"""The WASI backend (a spike): ``runner serve`` built for wasip1, run in-process by wasmtime.

wasmtime-py can send the guest's stdout to a callback but can only read
stdin from a file, so stdin is a FIFO that this side keeps open read-write
(so opening it never blocks and the guest sees EOF only when we close it).
The guest runs ``_start`` in a thread; wasmtime releases the GIL while it
runs.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import queue
import select
import shutil
import tempfile
import threading
from typing import Any

from ._process import WASM_ENV, decode_response
from .errors import RunnerError

_SHUTDOWN_TIMEOUT = 5.0

_lock = threading.Lock()
_engine: Any = None
_modules: dict[tuple[str, int, int], Any] = {}


def find_wasm(explicit: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Find runner.wasm: an explicit path, then $SCRIPTLET_TESTING_RUNNER_WASM."""
    if explicit is not None:
        path = pathlib.Path(explicit)
    elif os.environ.get(WASM_ENV):
        path = pathlib.Path(os.environ[WASM_ENV])
    else:
        raise RunnerError(
            f'no wasm runner: set ${WASM_ENV} to a runner built with '
            'GOOS=wasip1 GOARCH=wasm go build -o runner.wasm .'
        )
    if not path.is_file():
        raise RunnerError(f'wasm runner {path} does not exist')
    return path


def _module(path: pathlib.Path) -> tuple[Any, Any]:
    """One engine per Python process, and one compiled module per file version."""
    global _engine
    import wasmtime

    with _lock:
        if _engine is None:
            config = wasmtime.Config()
            # wasmtime's on-disk cache, so later processes skip compilation.
            config.cache = True
            _engine = wasmtime.Engine(config)
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        if key not in _modules:
            _modules[key] = wasmtime.Module.from_file(_engine, str(path))  # pyright: ignore[reportUnknownMemberType]
        return _engine, _modules[key]


class WasmProcess:
    """``runner serve`` in a wasmtime instance, on a thread of this process."""

    def __init__(self, wasm: pathlib.Path, charm: pathlib.Path):
        import wasmtime

        engine, module = _module(wasm)
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._partial = bytearray()
        self.exit_code: int | None = None
        self._tmp = tempfile.mkdtemp(prefix='scriptlet-wasm-')
        fifo = os.path.join(self._tmp, 'stdin')
        os.mkfifo(fifo)
        self._stdin = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        self._stderr_path = os.path.join(self._tmp, 'stderr')

        wasi = wasmtime.WasiConfig()
        wasi.argv = ['runner', 'serve']
        wasi.stdin_file = fifo
        wasi.stdout_custom = self._on_stdout
        wasi.stderr_file = self._stderr_path
        # The guest sees the charm (or the directory holding a .charm file)
        # at the same path, read-only, and nothing else.
        root = charm if charm.is_dir() else charm.parent
        wasi.preopen_dir(str(root), str(root), fs_mutable=False)
        self._store: Any = wasmtime.Store(engine)
        self._store.set_wasi(wasi)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self._start: Any = linker.instantiate(self._store, module).exports(self._store)['_start']
        self._thread = threading.Thread(target=self._run, name='scriptlet-wasm', daemon=True)
        self._thread.start()

    def _on_stdout(self, data: bytes) -> None:
        self._partial.extend(data)
        while (i := self._partial.find(b'\n')) >= 0:
            self._lines.put(bytes(self._partial[:i]))
            del self._partial[: i + 1]

    def _run(self) -> None:
        import wasmtime

        try:
            self._start(self._store)
            self.exit_code = 0
        except wasmtime.ExitTrap as e:
            self.exit_code = e.code
        except Exception as e:  # a trap: record it for the next request
            self.exit_code = -1
            self._partial.extend(f'\n{type(e).__name__}: {e}'.encode())
        finally:
            self._lines.put(None)

    @property
    def closed(self) -> bool:
        return self._closed

    def _stderr_tail(self) -> str:
        try:
            text = pathlib.Path(self._stderr_path).read_bytes()[-2000:]
        except OSError:
            return ''
        text = text.decode(errors='replace').strip()
        return f'\nrunner stderr:\n{text}' if text else ''

    def _write(self, data: bytes) -> None:
        # Non-blocking, so a guest that stops reading can't hang us on a full FIFO.
        view = memoryview(data)
        while view:
            if not self._thread.is_alive():
                raise RunnerError(
                    f'wasm runner exited (code {self.exit_code}){self._stderr_tail()}'
                )
            try:
                n = os.write(self._stdin, view)
            except BlockingIOError:
                select.select([], [self._stdin], [], 0.1)
                continue
            view = view[n:]

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        import json

        with self._lock:
            if self._closed:
                raise RunnerError('the runner has been closed')
            self._next_id += 1
            req_id = self._next_id
            line = json.dumps({**req, 'id': req_id}, separators=(',', ':')).encode() + b'\n'
            self._write(line)
            got = self._lines.get()
            if got is None:
                self._lines.put(None)
                got = b''
            return decode_response(got, req_id, self._stderr_tail())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._thread.is_alive():
                    with contextlib.suppress(RunnerError):
                        self._write(b'{"op":"shutdown"}\n')
                os.close(self._stdin)
                self._thread.join(_SHUTDOWN_TIMEOUT)
            finally:
                if not self._thread.is_alive():
                    # Drop the instance before the interpreter exits: wasmtime
                    # panics if its worker threads outlive the store.
                    self._start = None
                    self._store = None
                shutil.rmtree(self._tmp, ignore_errors=True)
