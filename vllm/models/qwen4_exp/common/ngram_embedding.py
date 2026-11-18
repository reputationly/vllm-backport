"""Alias: the pinned-memory Engram lookup kernel lives in the nvidia package
in this fork; upstream moved it to `common`."""
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (  # noqa: F401
    _lookup_ple_embedding_from_pinned_kernel,
)
