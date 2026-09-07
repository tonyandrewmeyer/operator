"""Retry policy on the live LLM seam.

Added after a 60-issue corpus run died partway through on an unhandled
`HTTP 429` (`spike-step-5/corpus-v2/RESULT.md`). Any run over more than a
handful of issues hits rate limiting, so this matters for Steps §6's GHA
rollout, not just local calibration.

No network: `urlopen` is monkeypatched and `sleep` is injected.
"""

import io
import json
import urllib.error

import pytest
from seams import llm
from seams.llm import DEFAULT_MAX_ATTEMPTS, LiveOpenRouterLLM, LLMError


def _response(payload: dict):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Resp(json.dumps(payload).encode())


def _http_error(code: int, retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("url", code, "boom", headers, None)


OK_PAYLOAD = {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"cost": 0.001}}


@pytest.fixture
def seam():
    slept = []
    client = LiveOpenRouterLLM(api_key="sk-test", sleep=slept.append)
    return client, slept


def test_retries_429_then_succeeds(monkeypatch, seam):
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) < 3:
            raise _http_error(429)
        return _response(OK_PAYLOAD)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert client.complete_json(purpose="extraction", prompt="p", context={}) == {"ok": True}
    assert len(calls) == 3
    assert slept == [2.0, 4.0]  # exponential from BACKOFF_BASE_S


def test_honours_retry_after_header(monkeypatch, seam):
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise _http_error(429, retry_after="7")
        return _response(OK_PAYLOAD)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client.complete_json(purpose="extraction", prompt="p", context={})
    assert slept == [7.0]


def test_unparseable_retry_after_falls_back_to_backoff(monkeypatch, seam):
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise _http_error(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        return _response(OK_PAYLOAD)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client.complete_json(purpose="extraction", prompt="p", context={})
    assert slept == [2.0]


@pytest.mark.parametrize("code", [401, 402, 404, 400])
def test_does_not_retry_permanent_failures(monkeypatch, seam, code):
    """A bad key or an exhausted balance will not fix itself.

    401 in particular: the first live attempt in this project burned a run on a
    revoked key, and retrying it five times with backoff would have wasted
    ~30s to reach the same answer.
    """
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise _http_error(code)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match=str(code)):
        client.complete_json(purpose="extraction", prompt="p", context={})
    assert len(calls) == 1
    assert slept == []


def test_gives_up_after_max_attempts(monkeypatch, seam):
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise _http_error(503)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError):
        client.complete_json(purpose="extraction", prompt="p", context={})
    assert len(calls) == DEFAULT_MAX_ATTEMPTS
    assert len(slept) == DEFAULT_MAX_ATTEMPTS - 1


def test_retries_network_errors(monkeypatch, seam):
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        return _response(OK_PAYLOAD)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client.complete_json(purpose="extraction", prompt="p", context={})
    assert len(calls) == 2


def test_invalid_json_is_not_retried(monkeypatch, seam):
    """A model that returns prose returns prose again -- retrying just costs money."""
    client, slept = seam
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _response({"choices": [{"message": {"content": "I think the bug is..."}}]})

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="valid JSON"):
        client.complete_json(purpose="extraction", prompt="p", context={})
    assert len(calls) == 1
    assert client.last_raw_content == "I think the bug is..."
