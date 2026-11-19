"""Torch implementations of the HYV4 Flash IndexPool preparation kernels.

The Triton kernels in `indexpool_prepare.py` hang inside Triton `_init_handles`
in the engine process on this A100 (sm80) host, even though they compile in
seconds in a clean process. The work is integer indexing over a handful of slots,
so do it in torch; the Triton path stays behind `VLLM_HY4_TRITON_PREP=1`.

Layouts follow the kernels exactly:
  * `coordinates` is [3, slots] (the caller slices row 0 for 1-D positions);
  * `raw` is [slots, ratio, dim];
  * entries of dead slots are left undefined by the kernel and are zeroed here.
"""

from __future__ import annotations

import torch


def _gather_rows(tensor, idx):
    """Index the first axis of a [tokens, dim] or [tokens, 1, dim] key tensor."""
    return tensor[idx] if tensor.dim() == 2 else tensor[idx, 0, :]

def _dbg(name, **kw):
    import os

    if os.environ.get("HY4_SHAPES"):
        print("HY4_SHAPES", name, *[f"{k}={tuple(v.shape) if hasattr(v, 'shape') else v}" for k, v in kw.items()], flush=True)

def _pos_of(positions, flat):
    """Positions of the given token indices as [N, 3].

    Accepts [3, tokens] (mRoPE planes), [tokens, 3] and 1-D [tokens].
    """
    if positions.dim() == 1:
        return positions[flat][:, None].expand(-1, 3)
    if positions.shape[0] == 3:
        return positions[:, flat].transpose(0, 1)
    if positions.shape[-1] == 3:
        return positions[flat]
    return positions[flat][:, :1].expand(-1, 3)
def _live_ends(starts: torch.Tensor, lengths: torch.Tensor, ratio: int):
    seg = starts[1:] - starts[:-1]
    context = lengths - seg
    live = (seg > 0) & (context >= 0)
    count = torch.where(
        live,
        lengths // ratio - torch.clamp(context, min=0) // ratio,
        torch.zeros_like(lengths),
    )
    return torch.cumsum(count, dim=0), context


def prepare_groups(keys, positions, tail, table, starts, lengths, tokens, ratio):
    device = keys.device
    _dbg("prepare_groups", keys=keys, positions=positions, tail=tail, table=table, starts=starts, lengths=lengths, tokens=tokens, ratio=ratio)
    starts = starts.reshape(-1).to(torch.int64)
    lengths = lengths.reshape(-1).to(torch.int64)
    batch = int(lengths.numel())
    slots = (tokens + batch * (ratio - 1)) // ratio
    dim = keys.shape[-1]
    ends, context = _live_ends(starts, lengths, ratio)
    # NOTE: never materialise `tail` here - under PP/packed layouts it is a
    # strided view over the whole KV buffer and `.contiguous()` would copy it.
    tail_c = tail
    ring = tail.shape[1]

    # ---- per slot: which request, which group, which members ----
    slot = torch.arange(slots, device=device, dtype=torch.int64)
    req = torch.searchsorted(ends, slot, right=True)  # first r with ends[r] > slot
    slot_live = req < batch
    req_c = torch.clamp(req, max=max(batch - 1, 0))

    start_s = starts[req_c]
    end_s = starts[req_c + 1]
    ctx_s = torch.clamp(lengths[req_c] - (end_s - start_s), min=0)
    prev = torch.where(
        req_c > 0,
        ends[torch.clamp(req_c - 1, min=0)],
        torch.zeros_like(ends[req_c]),
    )
    group = ctx_s // ratio + slot - prev
    first = group * ratio
    block = table.reshape(-1)[req_c].to(torch.int64)

    member = first[:, None] + torch.arange(ratio, device=device, dtype=torch.int64)
    old = member < ctx_s[:, None]
    source = start_s[:, None] + member - ctx_s[:, None]
    src_ok = (source >= 0) & (source < tokens)
    src_idx = torch.clamp(source, 0, max(tokens - 1, 0))

    retained = tail_c[block[:, None], member % ring, :dim]
    current = _gather_rows(keys, src_idx.reshape(-1)).reshape(slots, ratio, dim)
    raw = torch.where(old[:, :, None], retained, torch.where(
        src_ok[:, :, None], current, torch.zeros_like(current)
    )).to(keys.dtype)
    raw = torch.where(slot_live[:, None, None], raw, torch.zeros_like(raw))

    # ---- per group: the first member's mRoPE coordinates ----
    xyz = tail_c[block, first % ring, dim : dim + 3]
    source_first = start_s + first - ctx_s
    sf = torch.clamp(source_first, 0, max(tokens - 1, 0))
    pos = _pos_of(positions, sf)
    coordinates = torch.where((first < ctx_s)[:, None], xyz, pos.to(xyz.dtype))
    coordinates = torch.where(slot_live[:, None], coordinates, torch.zeros_like(coordinates))
    coordinates = coordinates.transpose(0, 1).contiguous()  # [3, slots]

    # ---- per token metadata ----
    tok = torch.arange(tokens, device=device, dtype=torch.int64)
    req_t = torch.clamp(
        torch.searchsorted(starts, tok, right=True) - 1, min=0, max=max(batch - 1, 0)
    )
    start_t = starts[req_t]
    end_t = starts[req_t + 1]
    ctx_t = lengths[req_t] - (end_t - start_t)
    live_t = (req_t < batch) & (end_t > start_t) & (ctx_t >= 0)
    logical = torch.clamp(ctx_t, min=0) + tok - start_t
    upper = torch.where(live_t, logical // ratio, torch.zeros_like(logical))

    group_positions = coordinates if positions.ndim == 2 else coordinates[0]
    return (
        raw.contiguous(),
        group_positions.contiguous(),
        req_c.to(torch.int32).contiguous(),
        group.contiguous(),
        slot_live.contiguous(),
        req_t.to(torch.int32).contiguous(),
        logical.contiguous(),
        upper.to(torch.int32).contiguous(),
        live_t.contiguous(),
    )


_COUNTS: dict[str, list] = {}


def _count(label: str) -> None:
    import time as _t

    slot = _COUNTS.setdefault(label, [0, _t.time()])
    slot[0] += 1
    now = _t.time()
    if now - slot[1] >= 15.0:
        print(
            f"HY4_RATE {label} {slot[0] / (now - slot[1]):.1f}/s",
            flush=True,
        )
        slot[0], slot[1] = 0, now


def _timed(label, fn, *a, **kw):
    import os
    import time as _t

    if not os.environ.get("HY4_TIMING"):
        return fn(*a, **kw)
    t0 = _t.time()
    out = fn(*a, **kw)
    dt = _t.time() - t0
    if dt > 0.05:
        print(f"HY4_TIMING {label} {dt*1000:.1f}ms", flush=True)
    return out


def write_tails(keys, positions, tail, table, starts, lengths) -> None:
    return _timed("write_tails", _write_tails_impl, keys, positions, tail, table, starts, lengths)


def _write_tails_impl(keys, positions, tail, table, starts, lengths) -> None:
    _count("write_tails")
    """Refresh the per-request key/position ring, vectorised (no Python loop).

    Mirrors `_write_tails`: for every (request, offset) the token index is
    `length - ring + offset`; tokens older than the live context are skipped and
    the rest are copied from `keys`/`positions` into `tail[block, token % ring]`.
    """
    _dbg("write_tails", keys=keys, positions=positions, tail=tail, table=table, starts=starts, lengths=lengths)
    starts = starts.reshape(-1).to(torch.int64)
    lengths = lengths.reshape(-1).to(torch.int64)
    batch = int(lengths.numel())
    if batch == 0:
        return
    device = keys.device
    ring = tail.shape[1]
    dim = keys.shape[-1]

    start_r = starts[:batch]  # `starts` carries one extra sentinel entry
    seg = starts[1:] - start_r
    context = lengths - seg
    req = torch.arange(batch, device=device, dtype=torch.int64)
    block = table.reshape(-1)[req].to(torch.int64)

    off = torch.arange(ring, device=device, dtype=torch.int64)
    token = lengths[:, None] - ring + off[None, :]
    live = (seg > 0)[:, None] & (context >= 0)[:, None] & (token >= context[:, None])
    source = start_r[:, None] + token - context[:, None]
    ok = live & (source >= 0) & (source < keys.shape[0])
    # NOTE: never materialise a boolean here (`bool(ok.any())`): it syncs the
    # stream and, under TP8, stalls the whole step behind pending collectives.
    src_idx = torch.clamp(source, 0, max(keys.shape[0] - 1, 0))

    rows = _gather_rows(keys, src_idx.reshape(-1)).reshape(batch, ring, dim)
    blk = block[:, None].expand(batch, ring)
    slot = token % ring
    tail[blk, slot, :dim] = torch.where(
        ok[:, :, None], rows.to(tail.dtype), tail[blk, slot, :dim]
    )

    pos = _pos_of(positions, src_idx.reshape(-1)).reshape(batch, ring, 3)
    tail[blk, slot, dim : dim + 3] = torch.where(
        ok[:, :, None], pos.to(tail.dtype), tail[blk, slot, dim : dim + 3]
    )
