# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

# Batches whose draft tokens are still waiting to be folded into the
# per-request table. With pipeline-parallel microbatching several batches are
# in flight; the scheduler only asks for drafts on the deferred structured
# output path, so entries are folded lazily and this bounds the backlog.
_MAX_PENDING_BATCHES = 16


class DraftTokensHandler:
    """Hands the draft tokens back to the scheduler for grammar validation.

    Drafts are kept **per request** rather than only for the last batch. With
    pipeline parallelism several microbatches are in flight and the scheduler
    asks for the drafts of a *deferred* batch right after a *different* batch
    finished (see `EngineCore.step_with_batch_queue`). Returning only the last
    batch left the deferred requests with their -1 placeholders: the grammar
    bitmask then skipped the draft rows and built the bonus row from the
    pre-draft grammar state, while the GPU verified the real drafts. Any bonus
    token sampled under that stale mask was rejected by the grammar and the
    request was terminated with 500.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)

        # Oldest first: (req_ids, draft_tokens_np | None, num_draft_tokens, event).
        self._pending: list[
            tuple[list[str], np.ndarray | None, int, torch.cuda.Event | None]
        ] = []
        # req_id -> latest draft token ids for that request.
        self._latest: dict[str, list[int]] = {}
        # Requests removed while a batch containing them was still pending.
        self._removed: set[str] = set()
        # Request ids of the most recent batch (the pre-existing contract of
        # get_draft_tokens() without arguments).
        self._last_req_ids: list[str] = []

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        if len(self._pending) >= _MAX_PENDING_BATCHES:
            self._drain()
        num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self._pending.append((input_batch.req_ids, None, num_draft_tokens, None))
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
            copy_event = torch.cuda.Event(blocking=True)
            copy_event.record()
        self._pending.append(
            (input_batch.req_ids, draft_tokens_np, num_draft_tokens, copy_event)
        )

    def remove_request(self, req_id: str) -> None:
        self._latest.pop(req_id, None)
        if self._pending:
            self._removed.add(req_id)

    def _drain(self) -> None:
        """Folds every pending batch into the per-request table."""
        for req_ids, draft_tokens_np, num_draft_tokens, copy_event in self._pending:
            if draft_tokens_np is not None:
                assert copy_event is not None
                copy_event.synchronize()
                rows = draft_tokens_np.tolist()
            else:
                # This case only happens when async scheduling is disabled.
                rows = [[-1] * num_draft_tokens for _ in req_ids]
            for req_id, row in zip(req_ids, rows):
                if req_id not in self._removed:
                    self._latest[req_id] = row
            self._last_req_ids = req_ids
        self._pending.clear()
        self._removed.clear()

    def get_draft_tokens(self, req_ids: list[str] | None = None) -> DraftTokenIds:
        """Returns the latest drafts of `req_ids` (default: the last batch)."""
        self._drain()
        if req_ids is None:
            req_ids = self._last_req_ids
        known = [req_id for req_id in req_ids if req_id in self._latest]
        return DraftTokenIds(known, [self._latest[req_id] for req_id in known])


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
