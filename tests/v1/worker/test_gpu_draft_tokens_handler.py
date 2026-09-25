# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DraftTokensHandler keeps drafts per request across in-flight batches.

Regression for structured output + speculative decoding + pipeline-parallel
microbatching: the scheduler asks for the drafts of a *deferred* batch after a
*different* batch finished. Handing back only the last batch left the deferred
requests with -1 placeholders, so the grammar bitmask built the bonus row from
the pre-draft state while the GPU verified the real drafts; the bonus token was
then rejected by the grammar and the request terminated with HTTP 500.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DraftTokensHandler uses CUDA streams"
)


def _batch(req_ids: list[str], structured: bool = True) -> SimpleNamespace:
    return SimpleNamespace(req_ids=req_ids, has_structured_output_reqs=structured)


def _drafts(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def test_deferred_batch_gets_its_own_drafts():
    handler = DraftTokensHandler(torch.device("cuda"))
    # Two microbatches in flight: r1's drafts were produced before r2's.
    handler.set_draft_tokens(_batch(["r1"]), _drafts([[1, 2, 3]]))
    handler.set_draft_tokens(_batch(["r2"]), _drafts([[4, 5, 6]]))

    got = handler.get_draft_tokens(["r1"])
    assert got.req_ids == ["r1"]
    assert got.draft_token_ids == [[1, 2, 3]]

    # The old contract (no argument) still means "the last batch".
    got = handler.get_draft_tokens()
    assert got.req_ids == ["r2"]
    assert got.draft_token_ids == [[4, 5, 6]]


def test_latest_drafts_win_and_unknown_requests_are_skipped():
    handler = DraftTokensHandler(torch.device("cuda"))
    handler.set_draft_tokens(_batch(["r1"]), _drafts([[1, 2, 3]]))
    handler.set_draft_tokens(_batch(["r1"]), _drafts([[7, 8, 9]]))

    got = handler.get_draft_tokens(["r1", "never-seen"])
    assert got.req_ids == ["r1"]
    assert got.draft_token_ids == [[7, 8, 9]]


def test_removed_request_is_forgotten_even_if_its_batch_was_pending():
    handler = DraftTokensHandler(torch.device("cuda"))
    handler.set_draft_tokens(_batch(["r1", "r2"]), _drafts([[1, 2, 3], [4, 5, 6]]))
    handler.remove_request("r1")

    got = handler.get_draft_tokens(["r1", "r2"])
    assert got.req_ids == ["r2"]
    assert got.draft_token_ids == [[4, 5, 6]]


def test_batches_without_structured_requests_yield_placeholders():
    handler = DraftTokensHandler(torch.device("cuda"))
    handler.set_draft_tokens(_batch(["r1"], structured=False), _drafts([[1, 2, 3]]))

    got = handler.get_draft_tokens()
    assert got.req_ids == ["r1"]
    assert got.draft_token_ids == [[-1, -1, -1]]


def test_pending_backlog_is_bounded():
    handler = DraftTokensHandler(torch.device("cuda"))
    for i in range(40):
        handler.set_draft_tokens(_batch([f"r{i}"]), _drafts([[i, i, i]]))
    # Never asked for drafts: the backlog must have been folded, not grown.
    assert len(handler._pending) < 40
    got = handler.get_draft_tokens(["r0", "r39"])
    assert got.draft_token_ids == [[0, 0, 0], [39, 39, 39]]
