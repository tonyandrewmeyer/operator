"""The runner process: one ``runner serve`` per context, newline-delimited JSON."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import tempfile
import threading
from typing import IO, Any, Protocol, cast

from .errors import RunnerError

# Environment variable naming the runner binary (or, for the wasm backend,
# the .wasm module).
RUNNER_ENV = 'SCRIPTLET_TESTING_RUNNER'
WASM_ENV = 'SCRIPTLET_TESTING_RUNNER_WASM'

# The protocol version this package speaks (the runner's ``protocol``).
PROTOCOL = 1

# How long close() waits for the runner to exit before killing it.
_SHUTDOWN_TIMEOUT = 5.0


def find_runner(explicit: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Find the native runner: an explicit path, then $SCRIPTLET_TESTING_RUNNER, then bundled."""
    if explicit is not None:
        path = pathlib.Path(explicit)
    elif os.environ.get(RUNNER_ENV):
        path = pathlib.Path(os.environ[RUNNER_ENV])
    else:
        path = pathlib.Path(__file__).parent / '_bin' / 'runner'
        if not path.exists():
            raise RunnerError(
                f'no scriptlet runner: set ${RUNNER_ENV} to a runner built from runner/ '
                '(go build -o runner .), or install a wheel that bundles one'
            )
    if not path.is_file():
        raise RunnerError(f'scriptlet runner {path} does not exist')
    return path


class Backend(Protocol):
    """A running ``serve`` loop that answers one request at a time."""

    def request(self, req: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...


def decode_response(line: bytes, req_id: int, stderr: str) -> dict[str, Any]:
    """Parse one response line and check it answers ``req_id``."""
    if not line:
        raise RunnerError(f'runner exited without answering request {req_id}{stderr}')
    try:
        resp = json.loads(line)
    except ValueError as e:
        raise RunnerError(f'runner sent invalid JSON: {line[:200]!r}{stderr}') from e
    if not isinstance(resp, dict):
        raise RunnerError(f'runner sent a non-object response: {line[:200]!r}')
    resp = cast('dict[str, Any]', resp)
    if resp.get('id') != req_id:
        raise RunnerError(f'runner answered request {resp.get("id")!r}, expected {req_id}')
    return resp


class NativeProcess:
    """``runner serve`` as a subprocess, over pipes."""

    def __init__(self, runner: pathlib.Path):
        self._lock = threading.Lock()
        self._next_id = 0
        # A file, not a pipe, so a chatty runner can't fill a pipe nobody reads.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 (closed in close())
        try:
            self._proc = subprocess.Popen(
                [str(runner), 'serve'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
            )
        except OSError as e:
            self._stderr.close()
            raise RunnerError(f'cannot start scriptlet runner {runner}: {e}') from e
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _stderr_tail(self) -> str:
        try:
            self._stderr.seek(0)
            text = self._stderr.read()[-2000:].decode(errors='replace').strip()
        except (OSError, ValueError):
            return ''
        return f'\nrunner stderr:\n{text}' if text else ''

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RunnerError('the runner has been closed')
            self._next_id += 1
            req_id = self._next_id
            stdin = cast('IO[bytes]', self._proc.stdin)
            stdout = cast('IO[bytes]', self._proc.stdout)
            line = json.dumps({**req, 'id': req_id}, separators=(',', ':')).encode() + b'\n'
            try:
                stdin.write(line)
                stdin.flush()
            except OSError as e:
                raise RunnerError(f'cannot write to the runner: {e}{self._stderr_tail()}') from e
            return decode_response(stdout.readline(), req_id, self._stderr_tail())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            try:
                if proc.poll() is None:
                    stdin = cast('IO[bytes]', proc.stdin)
                    try:
                        stdin.write(b'{"op":"shutdown"}\n')
                        stdin.flush()
                    except OSError:
                        pass
                    with contextlib.suppress(OSError):
                        stdin.close()
                    try:
                        proc.wait(timeout=_SHUTDOWN_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            finally:
                for f in (proc.stdin, proc.stdout):
                    if f is not None:
                        with contextlib.suppress(OSError):
                            f.close()
                self._stderr.close()

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.poll()
