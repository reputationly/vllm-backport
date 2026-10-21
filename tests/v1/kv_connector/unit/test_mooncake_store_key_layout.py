# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake keys must encode the store's KV-cache group projection.

The key embeds the group's index in the store projection (``@group:{i}``), and
different deployments may project a different subset of the KV cache groups
(e.g. excluding the compressor ring, or the SWA groups under SWA bounded
replay). The namespace must therefore differ too, or a shared master can hand
one group's KV to another group.
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    KeyMetadata,
    PoolKey,
    store_group_layout_signature,
)


def _config(
    group_ids: tuple[int, ...], replay_tokens: tuple[int, ...] = (0, 0, 0)
) -> SimpleNamespace:
    groups = [
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=128, prefix_replay_tokens=replay)
        )
        for replay in replay_tokens
    ]
    return SimpleNamespace(prefix_cacheable_group_ids=group_ids, kv_cache_groups=groups)


def _key(namespace: str, group_id: int = 1) -> str:
    metadata = KeyMetadata(
        model_name="model",
        tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
        group_id=group_id,
        store_namespace=namespace,
    )
    return PoolKey(metadata, "ab" * 16).to_string()


def test_signature_tracks_the_store_group_projection():
    without_ring = _config((1, 2), (0, 0, 128))
    with_replay = _config((1,), (0, 0, 128))
    assert store_group_layout_signature(without_ring) == (
        store_group_layout_signature(_config((1, 2), (0, 0, 128)))
    )
    assert store_group_layout_signature(without_ring) != (
        store_group_layout_signature(with_replay)
    )
    assert store_group_layout_signature(without_ring).startswith("store_groups:v2:")


def test_keys_from_different_layouts_never_collide():
    layout_a = "@" + store_group_layout_signature(_config((1, 2), (0, 0, 128)))
    layout_b = "@" + store_group_layout_signature(_config((1,), (0, 0, 128)))
    assert _key(layout_a) != _key(layout_b)
    # Same layout still produces identical keys (sharing within a layout).
    assert _key(layout_a) == _key(layout_a)
    # The group index itself is part of the key.
    assert _key(layout_a, group_id=1) != _key(layout_a, group_id=2)
