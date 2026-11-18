# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned pool pages and speculative tail rings for Flash IndexPool."""

from dataclasses import dataclass, replace
from math import lcm

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

POOL_PAGE_SIZE = 64


@dataclass
class IndexPoolMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int
    max_seq_len: int


class IndexPoolMetadataBuilder(AttentionMetadataBuilder[IndexPoolMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_model_len = vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm = common_attn_metadata
        return IndexPoolMetadata(
            query_start_loc=cm.query_start_loc[: cm.num_reqs + 1],
            seq_lens=cm.seq_lens[: cm.num_reqs],
            block_table=cm.block_table_tensor[: cm.num_reqs],
            num_actual_tokens=cm.num_actual_tokens,
            max_seq_len=cm.max_seq_len,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        """Reserve scoring width for replay beyond the dummy's short context."""
        metadata = self.build(0, common_attn_metadata)
        return replace(metadata, max_seq_len=self.max_model_len)


class IndexPoolBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ):
        """`ratio` tokens share one state, so a 256-token block is 64 pages."""
        ratio = int(
            get_current_vllm_config().model_config.hf_text_config.compressed_indexer_compress_ratio
        )
        states = max(1, block_size // ratio)
        return (num_blocks, states, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        """Kernel pages hold `POOL_PAGE_SIZE` states; `tokens_per_state` (=ratio)
        then makes the kv-manager block `ratio * POOL_PAGE_SIZE` tokens."""
        return [POOL_PAGE_SIZE]

    @classmethod
    def get_supported_head_sizes(cls):
        return [132]

    @classmethod
    def supports_pcp(cls):
        return False


class IndexPoolTailBackend(KpoolTailBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL_TAIL"

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Any layout: the ring is per-layer and addressed by explicit offsets.

        Declaring LBHNC only would forbid the block-outermost layout the mixed
        page sizes need (see `get_kv_cache_groups`).
        """
        return None

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False



def _mla_row_bytes(vllm_config) -> int:
    """Bytes per KV row of the main sparse-MLA cache (kv_lora_rank + rope)."""
    hf = vllm_config.model_config.hf_text_config
    return int(hf.kv_lora_rank) + int(hf.qk_rope_head_dim)


def _pad_page_to_mla_rows(page_bytes: int, row_bytes: int) -> int:
    return ((page_bytes + row_bytes - 1) // row_bytes) * row_bytes

def tail_ring_size(ratio: int, speculative_tokens: int) -> int:
    return ratio * next_power_of_2(cdiv(ratio + speculative_tokens, ratio))


class IndexPoolCache(DeepseekV32IndexerCache):
    def __init__(self, prefix, cache_config, ratio, *, tail=False):
        super().__init__(
            head_dim=131 if tail else 132,
            dtype=torch.float32 if tail else torch.uint8,
            prefix=prefix,
            cache_config=cache_config,
        )
        self.ratio = ratio
        self.tail = tail

    def get_attn_backend(self):
        return IndexPoolTailBackend if self.tail else IndexPoolBackend

    def get_kv_cache_config(self, vllm_config):
        ring = tail_ring_size(self.ratio, vllm_config.num_speculative_tokens)
        if self.tail:
            spec = AttentionSpec(
                block_size=ring,
                num_kv_heads=1,
                head_size=131,
                head_size_v=0,
                dtype=torch.float32,
                dcp_sharded=False,
                page_size_padded=_pad_page_to_mla_rows(
                    ring * 131 * 4, _mla_row_bytes(vllm_config)
                ),
            )
            return CircularBufferManager.layer_config(spec)
        spec = MLAAttentionSpec(
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
            block_size=lcm(
                self.cache_config.block_size, self.ratio * POOL_PAGE_SIZE, ring
            ),
            tokens_per_state=self.ratio,
            page_size_padded=_pad_page_to_mla_rows(
                lcm(
                    self.cache_config.block_size,
                    self.ratio * POOL_PAGE_SIZE,
                    ring,
                )
                // self.ratio
                * self.head_dim,
                _mla_row_bytes(vllm_config),
            ),
        )
        return FullAttentionManager.layer_config(spec)

    def bind_kv_cache(self, kv_cache):
        super().bind_kv_cache(kv_cache)
        if not self.tail and self.kv_cache.shape[1] != POOL_PAGE_SIZE:
            raise ValueError("IndexPool must be bound as 64-state kernel pages")
