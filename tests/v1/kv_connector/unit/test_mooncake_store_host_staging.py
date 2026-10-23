# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Mooncake host-staging transfer path."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_worker,
)


def test_split_by_capacity_keeps_keys_whole():
    sizes = [[10], [10], [10]]
    assert mooncake_worker._split_by_capacity(sizes, 25) == [(0, 2), (2, 3)]


def test_split_by_capacity_single_key_too_large():
    with pytest.raises(ValueError, match="exceeds staging capacity"):
        mooncake_worker._split_by_capacity([[30]], 20)


class _FakeStore:
    def __init__(self):
        self.put_calls = []
        self.get_calls = []

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        self.put_calls.append((list(keys), [list(a) for a in addrs]))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), [list(a) for a in addrs]))
        return [1] * len(keys)


class _FakePool:
    capacity_bytes = 20

    def __init__(self):
        self.committed = []

    def stage_out(self, addrs, sizes):
        return [[addr[0] + 1000] for addr in addrs]

    def stage_in(self, addrs, sizes):
        host_addrs = [[addr[0] + 1000] for addr in addrs]
        copies = [
            (addr[0], addr[0] + 1000, size[0])
            for addr, size in zip(addrs, sizes, strict=True)
        ]
        return host_addrs, copies

    def commit_in(self, copies):
        self.committed = list(copies)


def test_put_with_staging_chunks_and_rewrites_addrs():
    store = _FakeStore()
    pool = _FakePool()
    keys = ["k0", "k1", "k2"]
    addrs = [[10], [20], [30]]
    sizes = [[10], [10], [10]]

    assert mooncake_worker._batch_put_with_staging(
        store, pool, keys, addrs, sizes, config=object()
    ) == [0, 0, 0]

    # Capacity 20 fits two 10-byte keys, so the batch is split in two.
    assert [len(call[0]) for call in store.put_calls] == [2, 1]
    assert store.put_calls[0][1] == [[1010], [1020]]
    assert store.put_calls[1][1] == [[1030]]


def test_get_with_staging_commits_into_device():
    store = _FakeStore()
    pool = _FakePool()
    keys = ["k0", "k1"]
    addrs = [[10], [20]]
    sizes = [[10], [10]]

    assert mooncake_worker._batch_get_with_staging(store, pool, keys, addrs, sizes) == [
        1,
        1,
    ]

    assert store.get_calls[0][1] == [[1010], [1020]]
    assert pool.committed == [(10, 1010, 10), (20, 1020, 10)]


def test_without_staging_calls_store_directly():
    store = _FakeStore()
    keys = ["k0"]
    addrs = [[10]]
    sizes = [[10]]

    assert mooncake_worker._batch_put_with_staging(
        store, None, keys, addrs, sizes, config=object()
    ) == [0]
    assert store.put_calls[0][1] == [[10]]

    assert mooncake_worker._batch_get_with_staging(store, None, keys, addrs, sizes) == [
        1
    ]
    assert store.get_calls[0][1] == [[10]]
