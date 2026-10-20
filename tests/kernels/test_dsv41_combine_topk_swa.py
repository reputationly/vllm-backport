# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of the Torch and Triton ``combine_topk_swa_indices``.

The Ampere (SM8x) DSv4.1 path now uses the shared Triton kernel instead of the
Torch fallback.  The two must agree wherever the Torch version is defined
(``gather_len >= window``); when the gathered buffer is shorter than the
window (SWA bounded replay), only the Triton kernel's clamped semantics are
valid, so it is compared against a Python reference instead.

Ported with vllm-project/vllm#56638 (fb2ac82458) and the ``gather_start``
lower bound of #56227 (1dc2d854c1).
"""

import pytest
import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    _SPARSE_PREFILL_TOPK_ALIGNMENT,
    _combine_topk_swa_indices_torch,
)
from vllm.models.deepseek_v4_1.common.ops import combine_topk_swa_indices

WINDOW = 128


def _reference(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    window: int = WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    width = (
        (topk + window + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    indices = torch.full(
        (num_tokens, width), -1, dtype=torch.int32, device=topk_indices.device
    )
    lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)

    for req in range(seq_lens.shape[0]):
        qs = int(query_start_loc[req])
        qe = int(query_start_loc[req + 1])
        seq_len = int(seq_lens[req])
        gather_len = int(gather_lens[req])
        gather_start = seq_len - gather_len
        start_pos = seq_len - (qe - qs)
        for tok in range(qs, qe):
            pos = start_pos + (tok - qs)
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_start = max(pos - (window - 1), 0)
            swa_start = max(swa_start, gather_start)
            swa_len = pos - swa_start + 1
            indices[tok, :topk_len] = topk_indices[tok, :topk_len] + M * req
            if swa_len > 0:
                offsets = torch.arange(swa_start, pos + 1, device=indices.device)
                indices[tok, topk_len : topk_len + swa_len] = (
                    M * req + N + offsets - gather_start
                )
            lens[tok] = topk_len + swa_len
    return indices, lens


def _make_inputs(
    *,
    num_reqs: int,
    query_lens: list[int],
    prefix_lens: list[int],
    gather_lens: list[int],
    topk: int,
    M: int,
    N: int,
    seed: int,
    device: str,
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    total_q = int(query_start_loc[-1])
    seq_lens = torch.tensor(
        [q + p for q, p in zip(query_lens, prefix_lens)], dtype=torch.int32
    )
    gather_lens_t = torch.tensor(gather_lens, dtype=torch.int32)
    topk_indices = torch.randint(
        0,
        N,
        (total_q, topk),
        generator=gen,
        dtype=torch.int32,
    ).to(device)
    return (
        topk_indices,
        query_start_loc.to(device),
        seq_lens.to(device),
        gather_lens_t.to(device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
@pytest.mark.parametrize("topk", [64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_and_triton_agree_when_gather_covers_window(compress_ratio, topk, seed):
    num_reqs = 3
    query_lens = [1, 4, 7]
    prefix_lens = [0, 2 * WINDOW + 17, WINDOW]
    seq_lens_list = [q + p for q, p in zip(query_lens, prefix_lens)]
    gather_lens = [
        min(seq_lens_list[0], WINDOW),
        min(seq_lens_list[1], WINDOW + 5),
        seq_lens_list[2],
    ]
    M, N = 96, 4096
    topk_indices, qsl, seqlens, gathers = _make_inputs(
        num_reqs=num_reqs,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        gather_lens=gather_lens,
        topk=topk,
        M=M,
        N=N,
        seed=seed,
        device="cuda",
    )

    ref_indices, ref_lens = _combine_topk_swa_indices_torch(
        topk_indices, qsl, seqlens, gathers, WINDOW, compress_ratio, topk, M, N
    )
    tri_indices, tri_lens = combine_topk_swa_indices(
        topk_indices, qsl, seqlens, gathers, WINDOW, compress_ratio, topk, M, N
    )

    torch.testing.assert_close(tri_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        tri_indices[:, : topk + WINDOW], ref_indices[:, : topk + WINDOW], rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compress_ratio", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triton_clamps_window_to_replay_start(compress_ratio, seed):
    num_reqs = 4
    query_lens = [1, 2, 5, 8]
    prefix_lens = [0, WINDOW // 2, WINDOW * 2, 3 * WINDOW]
    # Deliberately shorter than the window: SWA bounded replay semantics.
    gather_lens = [1, 7, WINDOW // 2, WINDOW - 3]
    M, N = 128, 8192
    topk = 128
    topk_indices, qsl, seqlens, gathers = _make_inputs(
        num_reqs=num_reqs,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        gather_lens=gather_lens,
        topk=topk,
        M=M,
        N=N,
        seed=seed + 10,
        device="cuda",
    )

    tri_indices, tri_lens = combine_topk_swa_indices(
        topk_indices, qsl, seqlens, gathers, WINDOW, compress_ratio, topk, M, N
    )
    ref_indices, ref_lens = _reference(
        topk_indices, qsl, seqlens, gathers, compress_ratio, topk, M, N
    )

    torch.testing.assert_close(tri_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        tri_indices[:, : topk + WINDOW], ref_indices[:, : topk + WINDOW], rtol=0, atol=0
    )
