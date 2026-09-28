# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NoPE (head_dim 512) sparse MLA on the Triton kernel (GLM-5.3-Flash sm8x)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-alike GPU"
)


def _ref(q, kv, idx, sm_scale, dv=512):
    # q [T,H,D], kv [N,1,D], idx [T,1,K]
    T, H, _ = q.shape
    out = torch.zeros(T, H, dv, dtype=torch.float32, device=q.device)
    for t in range(T):
        sel = idx[t, 0]
        valid = (sel >= 0) & (sel < kv.shape[0])
        s = sel[valid].long()
        if s.numel() == 0:
            continue
        k = kv[s, 0].float()  # [k, D]
        qk = (q[t].float() @ k.T) * sm_scale  # [H, k]
        p = torch.softmax(qk, dim=-1)
        out[t] = p @ k[:, :dv]
    return out


@pytest.mark.parametrize("dim_qk", [576, 512])
@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_triton_mla_sparse_nope_and_rope(dim_qk: int, num_kv_splits: int):
    torch.manual_seed(0)
    dev = "cuda"
    T, H, N, K = 4, 16, 2048, 128
    q = torch.randn(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(N, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    idx = torch.stack([torch.randperm(N, device=dev)[:K].int() for _ in range(T)])
    idx = idx.view(T, 1, K)
    sm = dim_qk**-0.5
    out = triton_mla_sparse_attention(
        q, kv, idx, sm_scale=sm, num_kv_splits=num_kv_splits
    )
    ref = _ref(q, kv, idx, sm)
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    assert rel < 2e-2, f"dim_qk={dim_qk} splits={num_kv_splits} rel={rel:.2e}"


@pytest.mark.parametrize("dim_qk", [576, 512])
@pytest.mark.parametrize("num_kv_splits", [1, 4])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
def test_triton_mla_sparse_fp8_kv(dim_qk: int, num_kv_splits: int, kv_scale: float):
    """Per-tensor-fp8 KV (uint8 rows + LUT decode) matches the quantized bf16
    reference: the comparison baseline is the fp8-quantized KV itself, so the
    only tolerance is the bf16 LUT decode (exact) vs the dot-precision noise.
    """
    torch.manual_seed(0)
    dev = "cuda"
    T, H, N, K = 4, 16, 2048, 128
    q = torch.randn(T, H, dim_qk, dtype=torch.bfloat16, device=dev)
    kv_bf16 = torch.randn(N, 1, dim_qk, dtype=torch.bfloat16, device=dev)
    # Quantize with the same per-tensor scale the write path uses: rows are
    # divided by kv_scale before e4m3fn encoding, so encode (kv / scale).
    kv_fp8 = (kv_bf16.float() / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    kv_u8 = kv_fp8.view(torch.uint8)
    idx = torch.stack([torch.randperm(N, device=dev)[:K].int() for _ in range(T)])
    idx = idx.view(T, 1, K)
    sm = dim_qk**-0.5

    out = triton_mla_sparse_attention(
        q,
        kv_u8,
        idx,
        sm_scale=sm,
        num_kv_splits=num_kv_splits,
        kv_scale=kv_scale,
    )
    # Reference: decode the same fp8 bytes back to bf16 (what the LUT does
    # bit-exactly) and run the reference attention on the dequantized KV.
    kv_deq = kv_fp8.to(torch.bfloat16) * kv_scale
    ref = _ref(q, kv_deq, idx, sm)
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    assert rel < 2e-2, (
        f"dim_qk={dim_qk} splits={num_kv_splits} scale={kv_scale} rel={rel:.2e}"
    )
