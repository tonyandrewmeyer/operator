# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Length-prefixed framing for the persistent isolated-worker protocol.

The persistent worker (see :mod:`scenario._isolated_worker`) talks to the parent
test process over the worker's ``stdin`` / ``stdout`` pipes.  A charm's output is
arbitrary, so the request/response stream is *framed* rather than line-delimited:
each message is a 4-byte big-endian unsigned length header followed by that many
bytes of UTF-8 JSON.

Both helpers operate on **binary** streams (``proc.stdin`` / ``proc.stdout`` on
the parent side; ``sys.stdin.buffer`` / ``sys.stdout.buffer`` in the worker).
"""

from __future__ import annotations

import struct
from typing import IO

_HEADER = struct.Struct('>I')


def write_frame(stream: IO[bytes], data: bytes) -> None:
    """Write a single length-prefixed frame to *stream* and flush it."""
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def _read_exact(stream: IO[bytes], count: int) -> bytes | None:
    """Read exactly *count* bytes from *stream*, or ``None`` on EOF.

    ``BinaryIO.read`` may return short reads on a pipe, so the loop keeps
    reading until *count* bytes have arrived or the stream closes.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def read_frame(stream: IO[bytes]) -> bytes | None:
    """Read a single length-prefixed frame from *stream*.

    Returns the frame body, or ``None`` if the stream reached EOF before a full
    frame could be read (which the parent treats as a worker crash).
    """
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    return _read_exact(stream, length)
