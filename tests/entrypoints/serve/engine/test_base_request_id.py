# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from starlette.requests import Request

from vllm.entrypoints.serve.engine.serving import BaseServing


def _req(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )


def test_oneapi_header_only():
    req = _req({"X-Oneapi-Request-Id": "rid-1"})
    assert BaseServing._base_request_id(req) == "rid-1"


def test_oneapi_header_wins_over_x_request_id():
    req = _req({"X-Oneapi-Request-Id": "rid-1", "X-Request-Id": "envoy-uuid"})
    assert BaseServing._base_request_id(req) == "rid-1"


def test_x_request_id_still_works():
    req = _req({"X-Request-Id": "xr-1"})
    assert BaseServing._base_request_id(req) == "xr-1"


def test_blank_oneapi_header_falls_back_to_x_request_id():
    req = _req({"X-Oneapi-Request-Id": "  ", "X-Request-Id": "xr-1"})
    assert BaseServing._base_request_id(req) == "xr-1"


def test_default_used_without_headers():
    req = _req({})
    assert BaseServing._base_request_id(req, default="body-id") == "body-id"


def test_random_uuid_without_headers_or_default():
    # random_uuid() 是 16 位 hex（MASK_64_BITS），这里只锁「返回了十六进制串」。
    req_id = BaseServing._base_request_id(_req({}))
    assert req_id
    assert all(c in "0123456789abcdef" for c in req_id)


def test_no_raw_request_uses_default():
    assert BaseServing._base_request_id(None, default="d") == "d"


def test_lowercase_header_name():
    req = _req({"x-oneapi-request-id": "rid-1"})
    assert BaseServing._base_request_id(req) == "rid-1"
