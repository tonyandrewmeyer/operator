"""The LLM seam (PLAN.md Approach §3 extraction call + surface-inference pass).

Two implementations:

- `LiveOpenRouterLLM` — calls OpenRouter with a key from `OPENROUTER_API_KEY`
  or `~/.config/explain-my-model.key` (PLAN.md Approach §8). Not exercised by
  the test suite (no network, no key there), but no longer untested overall:
  first live calls landed 2026-07-27, see `spike-step-5/live-llm/RESULT.md`.
  Real code path for the VM dry-run (Steps §5) and GHA rollout (Steps §6).
- `FixtureLLM` — replays recorded step-2 extraction / surface-inference
  outputs from `fixtures/`. What the pipeline actually runs against here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = "deepseek/deepseek-chat"  # PLAN.md Approach §8: OPENROUTER_MODEL default
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Transient statuses worth retrying, same set the explain-my-model CLI uses
# against this provider. 401/402/404 are not here: a bad key or an exhausted
# balance will not fix itself.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0

# This project's own key. The env-then-keyfile resolution order follows the
# explain-my-model CLI (`cli/explain_my_model_cli/llm.py`), but the key is
# separate: that project's key is revoked, and a per-project key keeps this
# one's spend attributable once Steps §6 starts calling OpenRouter from CI.
KEY_FILE = Path.home() / ".config" / "ai-reproducer.key"

# purpose -> fixtures subdirectory
_FIXTURE_DIRS = {
    "extraction": "extractions",
    "surface_inference": "surface",
    "composition": "compositions",
    "test_file_synthesis": "test_file_synthesis",
    # PLAN.md Approach §3 delta (`inscope_second_pass.py`): fires once per
    # first-pass drop, keyed by the same issue_number as "extraction".
    "inscope_second_pass": "inscope_second_pass",
}


class LLMSeam(Protocol):
    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        """Return a parsed JSON object matching the caller's schema.

        `purpose` identifies the call site ("extraction" | "surface_inference"),
        used by fixture implementations to pick a recording and by live
        implementations only for logging/telemetry. `context` carries
        call-identifying data (at minimum `issue_number`) that fixture
        implementations use to select a recording; live implementations
        ignore it beyond optional logging.
        """
        ...


class LLMError(RuntimeError):
    pass


def _resolve_api_key(explicit: str | None = None) -> str:
    """First hit wins: argument, env var, then `KEY_FILE`."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    if KEY_FILE.is_file():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise LLMError(
        f"No API key found. Set OPENROUTER_API_KEY or write the key to {KEY_FILE} "
        "(see harness/README.md), or use FixtureLLM for a dry run with no key."
    )


class LiveOpenRouterLLM:
    """Real OpenRouter implementation.

    Not covered by the test suite (which has neither network nor key), but
    exercised for real against the step-5 corpus on 2026-07-27 — see
    `spike-step-5/live-llm/RESULT.md`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep=time.sleep,
    ):
        self.api_key = _resolve_api_key(api_key)
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.max_attempts = max_attempts
        self._sleep = sleep  # injectable so tests don't actually wait

    def _post(self, body: bytes) -> dict:
        """POST with bounded retry on the transient codes.

        Added after a 60-issue corpus run died on an unhandled `HTTP 429`
        partway through (`spike-step-5/corpus-v2/RESULT.md`). Any run over
        more than a handful of issues will hit rate limiting, so this matters
        for Steps §6's GHA rollout, not just for local calibration. Mirrors
        the retry policy the explain-my-model CLI already uses against the
        same provider (`cli/explain_my_model_cli/llm.py`).
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_attempts:
                    raise LLMError(f"OpenRouter returned HTTP {exc.code}") from exc
                last_error = exc
                # Honour Retry-After when the server sends one; otherwise back
                # off exponentially from BACKOFF_BASE_S.
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else BACKOFF_BASE_S * 2 ** (attempt - 1)
                except ValueError:
                    delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
                self._sleep(delay)
            except urllib.error.URLError as exc:
                if attempt == self.max_attempts:
                    raise LLMError(f"OpenRouter unreachable: {exc.reason}") from exc
                last_error = exc
                self._sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
        raise LLMError(f"OpenRouter retries exhausted: {last_error}")  # pragma: no cover

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode()
        payload = self._post(body)
        content = payload["choices"][0]["message"]["content"]
        # Kept so calibration runs can record exactly what the model said,
        # including when it fails to parse (`spike-step-5/live-llm/`).
        self.last_raw_content = content
        self.last_usage = payload.get("usage")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{purpose}: model did not return valid JSON; got {content[:200]!r}"
            ) from exc


class FixtureLLM:
    """Replays recorded LLM outputs. No network, no key required."""

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = Path(fixtures_dir)

    def complete_json(self, *, purpose: str, prompt: str, context: dict) -> dict:
        subdir = _FIXTURE_DIRS.get(purpose)
        if subdir is None:
            raise LLMError(f"FixtureLLM has no recordings for purpose={purpose!r}")
        issue_number = context.get("issue_number")
        if issue_number is None:
            raise LLMError("FixtureLLM requires context['issue_number'] to select a recording")
        path = self.fixtures_dir / subdir / f"{issue_number}.json"
        if not path.exists():
            raise LLMError(
                f"no {purpose} fixture recorded for issue #{issue_number} (looked at {path})"
            )
        return json.loads(path.read_text())
