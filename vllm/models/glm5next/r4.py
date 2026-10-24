# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online R4 (down_proj-input) block-Hadamard rotation.

kunlun-compressor's R4 transform folds a block-diagonal Hadamard ``H`` into
every ``down_proj`` weight (``W <- W @ H.T``).  ``H`` acts on the SwiGLU
intermediate activation, which is non-linear, so it cannot be absorbed into
the upstream ``gate``/``up`` projections: the serving engine has to apply
``H @ inter`` online, after the activation and before ``down_proj`` quantizes
its input.

The parameters are recorded by the quantizer in the exported ``config.json``
under the top-level ``kunlun_online_transforms`` list (see
``layer_stream_rtn/executor.py::_write_online_transform_manifest``)::

    {
        "kind": "online_hadamard_r4",
        "applies_to": "down_proj_input",
        "matrix_type": "hadamard",
        "block_size": 128,
        "random_signs": false,
        "seed": 42,
        "dims": [2048, 12288],
        "entries": [
            {"name": "dense_mlp", "target_pattern": "...", "dim": 12288},
            {"name": "routed_experts", "target_pattern": "...", "dim": 2048},
            {"name": "shared_experts", "target_pattern": "...", "dim": 2048},
        ],
    }

``H`` is stored/rebuilt exactly as ``kunlun_compressor.rotation.hadamard``
does it: ``block_diag(H_0, ..., H_{n-1})`` with
``H_i = H_bs / sqrt(bs) @ diag(sign_i)``.  ``random_signs=false`` (pure
Sylvester) is the cross-engine reproducible default.

Because ``H`` is block diagonal, a tensor-parallel shard of the intermediate
dimension can be rotated locally: as long as the shard boundaries fall on
block boundaries (true for every dim this model exports), no collective is
needed and the result equals rotating the full vector before the all-reduce.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)

KIND = "online_hadamard_r4"


def _sylvester_hadamard(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(n, n)`` Hadamard matrix by Sylvester's recursion (n a power of two)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard block size must be a power of two, got {n}")
    h = torch.tensor([[1.0]], dtype=dtype, device="cpu")
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h


class BlockHadamard:
    """Block-diagonal normalized Hadamard, mirroring the offline builder."""

    def __init__(
        self,
        dim: int,
        block_size: int,
        *,
        random_signs: bool,
        seed: int,
    ) -> None:
        if dim % block_size != 0:
            raise ValueError(
                f"R4 dim={dim} is not a multiple of block_size={block_size}"
            )
        n_blocks = dim // block_size
        h = _sylvester_hadamard(block_size) / math.sqrt(block_size)
        if random_signs:
            # Must match kunlun_compressor.rotation.hadamard.build_rotation
            # exactly: CPU generator, int8 draws, per-block sign vector.
            gen = torch.Generator().manual_seed(seed)
            signs = (
                torch.randint(
                    0,
                    2,
                    (n_blocks, block_size),
                    generator=gen,
                    dtype=torch.int8,
                    device="cpu",
                )
                * 2
                - 1
            ).to(torch.float32)
        else:
            signs = torch.ones(n_blocks, block_size, dtype=torch.float32, device="cpu")
        # blocks[i] = H / sqrt(bs) @ diag(sign_i)
        self.blocks = (h.unsqueeze(0) * signs.unsqueeze(1)).contiguous()
        self.dim = dim
        self.block_size = block_size
        self.random_signs = random_signs
        self.seed = seed
        self._device_cache: dict[torch.device, torch.Tensor] = {
            self.blocks.device: self.blocks
        }

    def _blocks_on(self, device: torch.device) -> torch.Tensor:
        cached = self._device_cache.get(device)
        if cached is None:
            cached = self.blocks.to(device, non_blocking=True).contiguous()
            self._device_cache[device] = cached
        return cached

    def apply(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        """Return ``H @ x`` for the local slice ``x[..., offset:offset+len]``.

        ``x`` is interpreted with the rotation dim on its last axis.  The
        returned tensor has the same shape/dtype; accumulation is fp32 to
        match the offline ``W @ H.T`` fold.
        """
        bs = self.block_size
        local = x.shape[-1]
        if local % bs != 0:
            raise RuntimeError(
                f"R4 local intermediate dim {local} is not a multiple of "
                f"block_size {bs}"
            )
        if offset % bs != 0:
            raise RuntimeError(
                f"R4 TP shard offset {offset} is not block aligned ({bs})"
            )
        n_local = local // bs
        start = offset // bs
        blocks = self._blocks_on(x.device)[start : start + n_local]
        if blocks.shape[0] != n_local:
            raise RuntimeError(
                f"R4 shard [{offset}:{offset + local}] exceeds dim {self.dim}"
            )
        orig_dtype = x.dtype
        xf = x.to(torch.float32).reshape(-1, n_local, bs)
        out = torch.einsum("rnb,ntb->rnt", xf, blocks)
        return out.reshape(x.shape).to(orig_dtype)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_ROTATIONS: dict[int, BlockHadamard] = {}
_ROLE_DIMS: dict[str, int] = {}
_CONFIG_SIGNATURE: str | None = None
# True when the manifest used the GLM-5.3 style `entries` list.  The
# single-dimension (`dim` + `target_pattern`) Kimi-K3 style manifest has no
# role names, so role lookups fall back to that one dim.
_HAS_ENTRIES = False


def configure(hf_config) -> bool:
    """Parse ``kunlun_online_transforms`` from the HF config (idempotent).

    Returns True when at least one R4 transform is active.  Safe to call
    from every layer constructor; the matrices are only built when the
    manifest changes.
    """
    global _CONFIG_SIGNATURE, _HAS_ENTRIES

    if isinstance(hf_config, dict):
        spec = hf_config.get("kunlun_online_transforms")
    else:
        spec = getattr(hf_config, "kunlun_online_transforms", None)
    if not spec:
        return bool(_ROTATIONS)

    transforms = [spec] if isinstance(spec, dict) else list(spec)
    signature = repr(transforms)
    if signature == _CONFIG_SIGNATURE:
        return bool(_ROTATIONS)

    rotations: dict[int, BlockHadamard] = {}
    role_dims: dict[str, int] = {}
    has_entries = False
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        kind = transform.get("kind")
        if kind != KIND:
            logger.warning_once(
                "Ignoring unsupported kunlun online transform kind=%r", kind
            )
            continue
        applies_to = transform.get("applies_to")
        if applies_to not in (None, "down_proj_input"):
            raise ValueError(
                f"Unsupported R4 applies_to={applies_to!r}; expected 'down_proj_input'"
            )
        matrix_type = transform.get("matrix_type", "hadamard")
        if matrix_type != "hadamard":
            raise ValueError(
                f"Unsupported R4 matrix_type={matrix_type!r}; only 'hadamard'"
            )

        block_size = int(transform.get("block_size", 128))
        seed = int(transform.get("seed", 42))
        random_signs = bool(transform.get("random_signs", True))

        dims: list[int] = []
        for entry in transform.get("entries") or []:
            has_entries = True
            dim = int(entry["dim"])
            dims.append(dim)
            name = entry.get("name")
            if name:
                role_dims[name] = dim
        if transform.get("dim") is not None:
            dims.append(int(transform["dim"]))
        dims.extend(int(d) for d in (transform.get("dims") or []))
        if not dims:
            raise ValueError(
                "R4 transform declares neither 'entries', 'dim' nor 'dims'"
            )

        for dim in sorted(set(dims)):
            rotations[dim] = BlockHadamard(
                dim,
                block_size,
                random_signs=random_signs,
                seed=seed,
            )

    if not rotations:
        return bool(_ROTATIONS)

    _ROTATIONS.update(rotations)
    _ROLE_DIMS.update(role_dims)
    _HAS_ENTRIES = has_entries
    _CONFIG_SIGNATURE = signature
    logger.info(
        "R4 online Hadamard enabled: dims=%s block_size=%s random_signs=%s seed=%s",
        sorted(rotations),
        sorted({r.block_size for r in rotations.values()}),
        sorted({r.random_signs for r in rotations.values()}),
        sorted({r.seed for r in rotations.values()}),
    )
    return True


def enabled() -> bool:
    return bool(_ROTATIONS)


def rotation_for_dim(dim: int) -> BlockHadamard | None:
    """Rotation folded into a weight whose input width is ``dim``, if any."""
    return _ROTATIONS.get(int(dim))


def role_dim(role: str) -> int | None:
    """Width declared for a manifest entry name (e.g. ``routed_experts``).

    Falls back to the single declared dim so a Kimi-K3 style one-dimension
    manifest keeps working.
    """
    if role in _ROLE_DIMS:
        return _ROLE_DIMS[role]
    if not _HAS_ENTRIES and len(_ROTATIONS) == 1:
        return next(iter(_ROTATIONS))
    return None


# --------------------------------------------------------------------------- #
# MoE hook
# --------------------------------------------------------------------------- #


def _local_offset(dim: int, local: int) -> int:
    """Element offset of this rank's contiguous TP shard."""
    if local == dim:
        return 0
    if dim % local != 0:
        raise RuntimeError(
            f"R4 dim {dim} is not divisible by the local MoE width {local}"
        )
    return get_tensor_model_parallel_rank() * local


def _iter_experts(experts) -> Iterable:
    """Yield concrete expert implementations (unwraps runtime fallbacks)."""
    if experts is None:
        return
    inner = getattr(experts, "experts", None)
    fallback = getattr(experts, "fallback_experts", None)
    if inner is not None and fallback is not None:
        yield from _iter_experts(inner)
        yield from _iter_experts(fallback)
    else:
        yield experts


def _hook_supported(experts) -> bool:
    """Whether ``apply`` routes the activation through ``self.activation``."""
    try:
        source = inspect.getsource(type(experts).apply)
    except (OSError, TypeError):
        return False
    return "self.activation" in source or "activation_func" in source


def install_moe_hook(runner, dim: int) -> int:
    """Apply ``H`` to every routed expert's activation output.

    Wraps the Python activation hook the modular MoE kernels expose between
    their two GEMMs, so ``H @ inter`` lands after the SwiGLU activation and
    before ``down_proj`` quantizes its input.  Returns the number of experts
    implementations hooked.
    """
    rotation = rotation_for_dim(dim)
    if rotation is None:
        raise RuntimeError(f"R4 requested for MoE dim {dim}, no rotation built")

    quant_method = getattr(
        getattr(runner, "routed_experts", None), "quant_method", None
    )
    if quant_method is None:
        raise RuntimeError(
            "R4: cannot locate the routed-experts quant method on the MoE "
            "runner; refusing to serve a rotated checkpoint silently"
        )
    if getattr(quant_method, "is_monolithic", False):
        raise RuntimeError(
            "R4: monolithic MoE kernels do not expose a down_proj input hook; "
            "use a modular backend (e.g. Marlin/Triton)"
        )

    kernel = getattr(quant_method, "moe_kernel", None)
    fused_experts = getattr(kernel, "fused_experts", None)
    if fused_experts is None:
        raise RuntimeError(
            f"R4: {type(quant_method).__name__} has no moe_kernel.fused_experts "
            "yet; install the hook after process_weights_after_loading"
        )

    installed = 0
    for experts in _iter_experts(fused_experts):
        if getattr(experts, "_vllm_r4_dim", None) is not None:
            continue
        if not _hook_supported(experts):
            raise RuntimeError(
                "R4: MoE backend "
                f"{type(experts).__name__} does not route its activation "
                "through self.activation(); cannot insert the online Hadamard"
            )

        orig_activation = experts.activation

        def activation(
            activation_enum,
            output,
            input,
            _orig=orig_activation,
            _dim=dim,
            **kwargs,
        ):
            _orig(activation_enum, output, input, **kwargs)
            rot = _ROTATIONS.get(_dim)
            if rot is None:
                return
            offset = _local_offset(_dim, output.shape[-1])
            output.copy_(rot.apply(output, offset=offset))

        experts.activation = activation
        experts._vllm_r4_dim = dim
        installed += 1

    if not installed:
        logger.warning_once("R4: routed-experts hook already installed for dim %s", dim)
    else:
        logger.info_once(
            "R4: installed online Hadamard on %d routed-experts backend(s) "
            "(dim=%d, block_size=%d)",
            installed,
            dim,
            rotation.block_size,
        )
    return installed


def install_moe_hooks(model) -> int:
    """Install the routed-experts hook on every R4-enabled MoE layer."""
    if not enabled():
        return 0
    total = 0
    seen: set[int] = set()
    for module in model.modules():
        dim = getattr(module, "_r4_routed_dim", None)
        if dim is None or id(module) in seen:
            continue
        seen.add(id(module))
        total += install_moe_hook(module.experts, dim)
    if total:
        logger.info_once("R4: routed-experts hook active on %d MoE layer(s)", total)
    return total
