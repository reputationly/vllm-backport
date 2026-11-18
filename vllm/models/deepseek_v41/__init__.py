"""Alias package: upstream spells it deepseek_v41, this fork deepseek_v4_1."""
import importlib as _importlib
import sys as _sys

_SUBMODULES = (
    "common",
    "common.engram",
    "common.ops",
    "nvidia",
    "nvidia.model_state",
    "attention",
    "sparse_mla",
    "compressor",
    "quant_config",
    "decoder_replay_layers",
)
for _name in _SUBMODULES:
    try:
        _sys.modules[f'{__name__}.{_name}'] = _importlib.import_module(
            f'vllm.models.deepseek_v4_1.{_name}'
        )
    except Exception:  # optional submodules differ between the two trees
        pass
