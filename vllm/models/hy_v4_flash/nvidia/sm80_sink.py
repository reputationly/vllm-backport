"""Learnable attention sink for HYV4 Flash on sm80 (Triton sparse MLA).

Upstream folds a per-head learned sink into FlashMLA by scaling each head's output
with ``exp(lse) / (exp(lse) + exp(sink))``. The Triton kernel did not expose
`lse`, so the sink used to be dropped (a different model, silently). This module
asks the kernel for `lse` and applies exactly that correction.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)


class HYV4SM80SinkImpl(TritonMLASparseImpl):
    """Triton sparse-MLA impl that applies HYV4's per-head learnable sink."""

    def __init__(self, *args, sinks: torch.Tensor | None = None, **kwargs) -> None:
        #  may forward the sink as a trailing positional extra arg.
        if sinks is None and args and isinstance(args[-1], torch.Tensor):
            sinks = args[-1]
            args = args[:-1]
        super().__init__(*args, **kwargs)
        # fp32 copy; the parameter is loaded per TP rank (padded heads = -inf).
        self.sinks = None if sinks is None else sinks.detach().to(torch.float32)

    def _apply_sink(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        if self.sinks is None:
            return out
        lse = lse.to(torch.float32)
        while lse.dim() > out.dim() - 1:
            lse = lse.squeeze(1) if lse.shape[1] == 1 else lse.reshape(out.shape[:2])
        lse = lse.reshape(out.shape[0], out.shape[1])
        sink = self.sinks[: out.shape[1]].to(out.device)
        # exp(lse) / (exp(lse) + exp(sink)), evaluated as a sigmoid in log space
        # so a -inf padded sink yields exactly zero contribution.
        scale = torch.sigmoid(lse - sink).unsqueeze(-1)
        return (out.to(torch.float32) * scale).to(out.dtype)

    def _forward_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        out, lse = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
            kv_scale=kv_scale,
            return_lse=True,
        )
        return self._apply_sink(out, lse)


class HYV4SM80SinkSparseBackend(TritonMLASparseBackend):
    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def get_impl_cls(cls):
        return HYV4SM80SinkImpl


class HYV4SM80SinkSWABackend(HYV4SM80SinkSparseBackend):
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True
