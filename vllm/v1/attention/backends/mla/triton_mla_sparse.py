# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Draft steps are single-token-per-request and uniform, so every field
        # this backend's forward reads is step-invariant: req_id_per_token is
        # arange over rows either way, block_table/query_start_loc come from
        # persistent engine buffers, and slot_mapping is refreshed in place by
        # the speculator's compute_slot_mappings before the update hook runs.
        # The one per-step device rebuild the XPU base does (np.repeat +
        # pinned H2D) is only needed for non-uniform query layouts. Like the
        # dense Triton builder, the update itself is a no-op; DCP keeps the
        # fallback because its local seq-lens are not advanced per step.
        self.dcp_world_size = 1
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            pass
        self.supports_draft_decode_metadata_update = self.dcp_world_size == 1

    def update_draft_decode_metadata(self, _metadata: XPUMLASparseMetadata) -> None:
        pass


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    # Per-tensor fp8 KV stays in its uint8 storage: the wrapper must not
    # reinterpret it as e4m3 bytes. `is_fp8_kv` in the kernel is a
    # `dtype == uint8` test, so an fp8-typed view would silently take the
    # bf16 path with a float8 pointer — which fails to compile below SM89
    # ("fp8e4nv not supported in this architecture").
    keeps_raw_kv_bytes: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        dim_qk = self.head_size
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        for kv in (
            # bf16 cache rows and, when enabled, per-tensor-fp8 (uint8) rows:
            # the IS_FP8_KV constexpr variants each autotune separately.
            torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device),
            torch.empty(64, 1, dim_qk, dtype=torch.uint8, device=device),
        ):
            if kv.dtype == torch.uint8 and not is_quantized_kv_cache(
                self.kv_cache_dtype
            ):
                continue
            indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
            for splits in KV_SPLITS_CANDIDATES:
                triton_mla_sparse_attention(
                    q,
                    kv,
                    indices,
                    sm_scale=self.softmax_scale,
                    num_kv_splits=splits,
                    sm_count=self._sm_count,
                )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Same orchestration as the XPU base, minus its blanket fp8 raise.

        Per-tensor fp8 KV stays in its uint8 storage — the model-side
        `_fp8_kv_needs_view` fp8 view is skipped for this backend (see
        deepseek_v32/attention.py), and the kernel decodes bytes through the
        SM80 LUT with the per-tensor scale folded back in. Query side stays
        bf16 (Mode 1): `supports_quant_query_input` is False.
        """
        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        buf = (
            self._indexer.topk_indices_buffer
            if self._indexer is not None
            else self.topk_indices_buffer
        )
        assert buf is not None, "topk_indices_buffer required for sparse MLA"
        topk_indices = buf[:num_actual_toks]

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            # The buffer's own width, not the logical top-k. GLM-5.3-Flash's
            # kpool indexer reserves `kpool - 1` extra slots for the in-progress
            # pool tail and rounds the total up to the sparse-MLA 128-column
            # tile (2048 -> 2176); the padding stays -1 and is masked. Models
            # whose buffer is exactly topk_tokens wide are unaffected.
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        kv_scale = (
            layer._k_scale_float if is_quantized_kv_cache(self.kv_cache_dtype) else 1.0
        )
        attn_out = self._forward_kv(
            q, kv_rows, topk_indices_global, attn_metadata, kv_scale
        )
        return attn_out, None

    def _forward_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
            kv_scale=kv_scale,
        )
        return output

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        return self._forward_kv(q, kv_c_and_k_pe_cache, topk_indices, attn_metadata)


class TritonMLASparseBackend(XPUMLASparseBackend):
    """Same sparse-MLA contract as the XPU backend, CUDA Triton kernels.

    Beyond bf16, this backend reads per-tensor-fp8 KV (`--kv-cache-dtype
    fp8`): uint8 rows decoded through the SM80 e4m3fn LUT with the
    per-tensor scale folded back in (dense TRITON_MLA's "Mode 1" — q stays
    bf16). fp8_ds_mla / nvfp4 layouts stay unsupported here.
    """

    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 latent + 64 RoPE (DeepSeek-V3.2 / GLM-5).
        # 512 = NoPE MLA (GLM-5.3-Flash, qk_rope_head_dim = 0).
        return [512, 576]

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl
