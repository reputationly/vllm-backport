# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the DeepSeek-V4.1 Engram host-lookup kernel before serving.

The lookup kernel is Triton-JIT'ed on first use; in production the first real
request paid the compile cost (``Triton kernel JIT compilation during
inference: _engram_lookup_kernel``). Run one lookup per table at startup so
the kernel is compiled before any request arrives.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def deepseek_v4_engram_warmup(model: torch.nn.Module) -> None:
    from vllm.models.deepseek_v4_1.common.engram import ParallelEngramEmbedding

    modules = [m for m in model.modules() if isinstance(m, ParallelEngramEmbedding)]
    if not modules:
        return

    device = torch.device("cuda", torch.accelerator.current_device_index())
    for module in modules:
        indices = torch.zeros(
            (1, module.part_n_hash_cols), dtype=torch.int32, device=device
        )
        out = torch.empty(
            (1, module.part_n_hash_cols, module.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        module.lookup(indices, out)

    logger.info_once(
        "Warmed up the DeepSeek-V4.1 Engram lookup kernel for %d table(s).",
        len(modules),
    )
