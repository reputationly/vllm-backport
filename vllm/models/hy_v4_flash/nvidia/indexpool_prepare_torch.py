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
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.compilation.breakable_cudagraph import eager_break_during_capture


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


_STEP_CACHE: dict[str, object] = {"key": None, "bundle": None, "pin": None}
_CALL_COUNTS: dict[str, list] = {}


def _count(label: str) -> None:
    """Print a per-15s call rate for `label` when HY4_TIMING is set."""
    import os
    import time as _t

    if not os.environ.get("HY4_TIMING"):
        return
    slot = _CALL_COUNTS.setdefault(label, [0, _t.time()])
    slot[0] += 1
    now = _t.time()
    if now - slot[1] >= 15.0:
        print(f"HY4_RATE {label} {slot[0] / (now - slot[1]):.1f}/s", flush=True)
        slot[0], slot[1] = 0, now


_CACHE_STATS = {"hit": 0, "miss": 0, "since": 0.0}


def _note_cache(hit: bool) -> None:
    import os
    import time as _t

    if not os.environ.get("HY4_CACHE_STATS"):
        return
    _CACHE_STATS["hit" if hit else "miss"] += 1
    now = _t.time()
    if _CACHE_STATS["since"] == 0.0:
        _CACHE_STATS["since"] = now
    elif now - _CACHE_STATS["since"] >= 15.0:
        h, m = _CACHE_STATS["hit"], _CACHE_STATS["miss"]
        total = h + m
        print(
            f"HY4_CACHE hit={h} miss={m} rate={h / total:.2f}",
            flush=True,
        )
        _CACHE_STATS.update(hit=0, miss=0, since=now)


def _step_index_math(starts_raw, lengths_raw, table_raw, positions, tokens, ratio, ring,
                     dim, device):
    """Index math shared by every sparse layer of one forward step.

    `ends/context/req/group/member/source/block` only depend on the scheduler's
    `query_start_loc`, `seq_lens`, `block_table` and the step's positions, so the
    six sparse layers of a stage would otherwise recompute the same ~20 small
    tensors six times. Keyed on the *original* tensor identities and pinned so an
    id cannot be recycled between steps, which makes a hit exactly "same step".
    """
    key = (
        id(starts_raw),
        id(lengths_raw),
        id(table_raw),
        id(positions),
        int(tokens),
        int(ratio),
        int(ring),
        int(dim),
    )
    # Under torch.compile the traced call would populate this cache with fake
    # tensors while keeping the same identity key, so the runtime call would
    # then read fake values. Compilation also removes the launch overhead this
    # cache exists to avoid, so it stays off there.
    compiling = torch.compiler.is_compiling()
    if not compiling and _STEP_CACHE["key"] == key:
        _note_cache(True)
        return _STEP_CACHE["bundle"]
    if not compiling:
        _note_cache(False)

    starts = starts_raw.reshape(-1).to(torch.int64)
    lengths = lengths_raw.reshape(-1).to(torch.int64)
    batch = int(lengths.numel())
    slots = (tokens + batch * (ratio - 1)) // ratio
    ends, context = _live_ends(starts, lengths, ratio)

    slot = torch.arange(slots, device=device, dtype=torch.int64)
    req = torch.searchsorted(ends, slot, right=True)
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
    block = table_raw.reshape(-1)[req_c].to(torch.int64)

    members = torch.arange(ratio, device=device, dtype=torch.int64)
    member = first[:, None] + members
    old = member < ctx_s[:, None]
    source = start_s[:, None] + member - ctx_s[:, None]
    src_ok = (source >= 0) & (source < tokens)
    src_idx = torch.clamp(source, 0, max(tokens - 1, 0))

    # coordinates fall back to the token positions for the group's first member
    source_first = start_s + first - ctx_s
    sf = torch.clamp(source_first, 0, max(tokens - 1, 0))
    pos_values = _pos_of(positions, sf).contiguous()
    use_xyz = (first < ctx_s)[:, None]

    tok = torch.arange(tokens, device=device, dtype=torch.int64)
    req_t = torch.clamp(
        torch.searchsorted(starts, tok, right=True) - 1,
        min=0,
        max=max(batch - 1, 0),
    )
    start_t = starts[req_t]
    end_t = starts[req_t + 1]
    ctx_t = lengths[req_t] - (end_t - start_t)
    live_t = (req_t < batch) & (end_t > start_t) & (ctx_t >= 0)
    logical = torch.clamp(ctx_t, min=0) + tok - start_t
    upper = torch.where(live_t, logical // ratio, torch.zeros_like(logical))

    bundle = {
        "slots": slots,
        "batch": batch,
        "member_row": (member % ring).reshape(-1),
        "member_sel": member % ring,
        "first_row": first % ring,
        "old": old,
        "src_ok": src_ok,
        "src_idx": src_idx.reshape(-1),
        "block": block,
        "use_xyz": use_xyz,
        "pos_values": pos_values,
        "slot_live": slot_live,
        "req_c": req_c,
        "group": group,
        "flat_positions": positions.ndim != 2,
        "req_t": req_t,
        "logical": logical,
        "upper": upper,
        "live_t": live_t,
    }
    if not compiling:
        _STEP_CACHE["key"] = key
        _STEP_CACHE["bundle"] = bundle
        _STEP_CACHE["pin"] = (starts_raw, lengths_raw, table_raw, positions)
    return bundle


def _prepare_groups_into(
    keys: torch.Tensor,
    positions: torch.Tensor,
    tail: torch.Tensor,
    table: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    tokens: int,
    ratio: int,
    raw: torch.Tensor,
    group_positions: torch.Tensor,
    requests: torch.Tensor,
    groups: torch.Tensor,
    group_live: torch.Tensor,
    owner: torch.Tensor,
    logical: torch.Tensor,
    upper: torch.Tensor,
    live: torch.Tensor,
) -> None:
    """Write the IndexPool index tensors into pre-allocated outputs.

    Registered as a custom op (see `prepare_groups`): the index maths mixes
    Python ints with `torch.searchsorted` and fancy indexing, which under
    torch.compile turns into symbolic shapes and produces out-of-bounds gather
    indices. Keeping it opaque makes it run eagerly on concrete integers.
    """
    (
        r,
        gp,
        rq,
        gr,
        gl,
        ow,
        lg,
        up,
        lv,
    ) = _prepare_groups_impl(keys, positions, tail, table, starts, lengths, tokens, ratio)
    raw.copy_(r)
    group_positions.copy_(gp)
    requests.copy_(rq)
    groups.copy_(gr)
    group_live.copy_(gl)
    owner.copy_(ow)
    logical.copy_(lg)
    upper.copy_(up)
    live.copy_(lv)


def _prepare_groups_into_fake(
    keys: torch.Tensor,
    positions: torch.Tensor,
    tail: torch.Tensor,
    table: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    tokens: int,
    ratio: int,
    raw: torch.Tensor,
    group_positions: torch.Tensor,
    requests: torch.Tensor,
    groups: torch.Tensor,
    group_live: torch.Tensor,
    owner: torch.Tensor,
    logical: torch.Tensor,
    upper: torch.Tensor,
    live: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="hy4_prepare_groups_into",
    op_func=_prepare_groups_into,
    mutates_args=[
        "raw",
        "group_positions",
        "requests",
        "groups",
        "group_live",
        "owner",
        "logical",
        "upper",
        "live",
    ],
    fake_impl=_prepare_groups_into_fake,
)


def prepare_groups(keys, positions, tail, table, starts, lengths, tokens, ratio):
    """Allocate the IndexPool index tensors and fill them through the custom op."""
    device = keys.device
    _dbg(
        "prepare_groups_impl", keys=keys, positions=positions, tail=tail, table=table,
        starts=starts, lengths=lengths, tokens=tokens, ratio=ratio,
    )
    slots = (tokens + lengths.numel() * (ratio - 1)) // ratio
    dim = keys.shape[-1]
    raw = keys.new_empty((slots, ratio, dim))
    group_positions = torch.empty(
        (3, slots) if positions.ndim == 2 else (slots,),
        dtype=torch.float32,
        device=device,
    )
    requests = torch.empty(slots, dtype=torch.int32, device=device)
    groups = torch.empty(slots, dtype=torch.int64, device=device)
    group_live = torch.empty(slots, dtype=torch.bool, device=device)
    owner = torch.empty(tokens, dtype=torch.int32, device=device)
    logical = torch.empty(tokens, dtype=torch.int64, device=device)
    upper = torch.empty(tokens, dtype=torch.int32, device=device)
    live = torch.empty(tokens, dtype=torch.bool, device=device)
    torch.ops.vllm.hy4_prepare_groups_into(
        keys, positions, tail, table, starts, lengths, tokens, ratio,
        raw, group_positions, requests, groups, group_live, owner, logical, upper, live,
    )
    return (
        raw,
        group_positions,
        requests,
        groups,
        group_live,
        owner,
        logical,
        upper,
        live,
    )


@eager_break_during_capture
def _prepare_groups_impl(keys, positions, tail, table, starts, lengths, tokens, ratio):
    device = keys.device
    _dbg(
        "prepare_groups", keys=keys, positions=positions, tail=tail, table=table,
        starts=starts, lengths=lengths, tokens=tokens, ratio=ratio,
    )
    dim = keys.shape[-1]
    ring = tail.shape[1]
    inv = _step_index_math(
        starts, lengths, table, positions, tokens, ratio, ring, dim, device
    )
    slots = inv["slots"]
    old, src_ok, slot_live = inv["old"], inv["src_ok"], inv["slot_live"]
    # Padding rows in a captured graph can carry stale ids; clamp so a replay
    # cannot index past the tail cache.
    block = inv["block"].clamp(0, tail.shape[0] - 1)

    retained = tail[block[:, None], inv["member_sel"], :dim]
    current = _gather_rows(keys, inv["src_idx"]).reshape(slots, ratio, dim)
    raw = torch.where(
        old[:, :, None],
        retained,
        torch.where(src_ok[:, :, None], current, torch.zeros_like(current)),
    ).to(keys.dtype)
    raw = torch.where(slot_live[:, None, None], raw, torch.zeros_like(raw))

    xyz = tail[block, inv["first_row"], dim : dim + 3]
    coordinates = torch.where(inv["use_xyz"], xyz, inv["pos_values"].to(xyz.dtype))
    coordinates = torch.where(
        slot_live[:, None], coordinates, torch.zeros_like(coordinates)
    )
    coordinates = coordinates.transpose(0, 1).contiguous()

    group_positions = coordinates if not inv["flat_positions"] else coordinates[0]
    return (
        raw.contiguous(),
        group_positions.contiguous(),
        inv["req_c"].to(torch.int32).contiguous(),
        inv["group"].contiguous(),
        slot_live.contiguous(),
        inv["req_t"].to(torch.int32).contiguous(),
        inv["logical"].contiguous(),
        inv["upper"].to(torch.int32).contiguous(),
        inv["live_t"].contiguous(),
    )


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


@eager_break_during_capture
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
    block = block.clamp(0, tail.shape[0] - 1)

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
