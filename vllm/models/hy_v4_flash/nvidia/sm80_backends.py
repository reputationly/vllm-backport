"""sm80 (A100) sparse-MLA backends for HYV4 Flash.

The upstream PR binds FlashMLA for both the sparse and the sliding-window
layers, which requires compute capability 9/10. On Ampere we bind the fork's
Triton sparse-MLA stack instead.

Step A (this file's current state): sink is *not* applied — the Triton kernel
returns no log-sum-exp, so the backend advertises `supports_sink() == False`
and the model must not pass `sinks=`. Numerics are therefore not the trained
architecture; Step B adds an LSE output to the kernel plus the
`out *= exp(lse) / (exp(lse) + exp(sink))` correction.
"""

from .sm80_sink import HYV4SM80SinkImpl
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
)


class HYV4SM80SparseBackend(TritonMLASparseBackend):
    """Sparse (IndexPool / DSA) layers on sm80."""

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def get_impl_cls(cls):
        return HYV4SM80SinkImpl


class HYV4SM80SWABackend(HYV4SM80SparseBackend):
    """Sliding-window layers on sm80: same kernel, window-sized index buffer."""

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True
