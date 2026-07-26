"""Message (de)serialization for the ZeroQueue transport.

Uses pickle protocol 5 (PEP 574), whose out-of-band buffer support both serializes
NumPy arrays (the bulk of a Frame) faster than the default protocol and lets the
transport move large pixel/audio/tensor buffers *beside* the pickle stream instead
of inside it. Keeping the codec in one place lets the mailbox serialize a message
**once** and hand the same bytes to every fan-out edge.

Wire framing
------------
Every message is a single ``bytes`` blob whose first byte is a tag:

* ``_TAG_PLAIN`` — the rest is one ordinary pickle (all buffers in-band). This is the
  default and the only format used unless shared memory is opted in.
* ``_TAG_SHM`` — the rest is a small pickle envelope: the metadata pickle plus an
  ordered list of POSIX shared-memory segment descriptors. Large out-of-band buffers
  travel through shared memory rather than over the socket.

Shared-memory transport (opt-in via ``NEUDC_SHM_IMAGES=1``)
----------------------------------------------------------
This is payload-agnostic: any object that exposes large NumPy arrays / bytes buffers
(``Frame.image``, ``Frame.embedding``, a segmentation mask, or a future audio/token
payload) benefits, because protocol 5 pulls those buffers out-of-band generically.

Ownership contract (single-consumer edges only — the mailbox enforces this and falls
back to ``_TAG_PLAIN`` for fan-out):

* Producer: create segment → copy bytes in → drop it from *this* process's
  ``resource_tracker`` (via ``track=False`` on 3.13+, else an explicit ``unregister``)
  → ``close()``. The segment stays alive; the producer never unlinks it.
* Consumer: open → **copy** into an owned ``bytearray`` → ``close()`` + ``unlink()``.

The copy on read is deliberate: nodes such as ``DrawNode`` mutate ``frame.image`` in
place, so a shared read-only view across processes would be unsafe. The ``bytearray``
copy is also writable, which those in-place mutations require.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from multiprocessing import resource_tracker, shared_memory
from typing import Any

from neudc.utils import LOGGER

# Protocol 5 (PEP 574): out-of-band buffers, efficient for large NumPy arrays.
_PROTOCOL = 5

# First byte of every framed message; tells the decoder how to read the rest.
_TAG_PLAIN = 0  # rest is a single in-band pickle
_TAG_SHM = 1  # rest is a pickle envelope referencing shared-memory segments

# Opt-in shared-memory transport for large buffers. Off by default.
SHM_ENABLED = os.environ.get("NEUDC_SHM_IMAGES", "0").lower() in {"1", "true", "yes", "on"}

# Only buffers at least this large are diverted to shared memory; smaller ones stay
# in-band, since a segment per tiny buffer would cost more than the copy it saves.
SHM_MIN_BYTES = int(os.environ.get("NEUDC_SHM_MIN_BYTES", "65536"))

# Python 3.13+ has SharedMemory(..., track=False) to skip resource_tracker entirely.
# On 3.11/3.12 the segment is always registered on create/attach, so we unregister by hand.
_SHM_HAS_TRACK = sys.version_info >= (3, 13)


def dumps(obj: Any, *, use_shm: bool = False) -> bytes:
    """Serialize a message object to a single tagged ``bytes`` blob.

    Args:
    ----
        obj: The message to serialize.
        use_shm: When True, large out-of-band buffers are moved through POSIX shared
            memory (see module docstring). The caller is responsible for only enabling
            this on single-consumer edges. When False, the classic all-in-band pickle
            is produced (byte-for-byte the historical format, minus the leading tag).

    """
    if use_shm:
        return _dumps_shm(obj)
    return bytes([_TAG_PLAIN]) + pickle.dumps(obj, protocol=_PROTOCOL)


def loads(raw: bytes) -> Any:
    """Deserialize bytes produced by :func:`dumps` back into an object."""
    view = memoryview(raw)
    tag = view[0]
    body = view[1:]
    if tag == _TAG_PLAIN:
        return pickle.loads(body)  # noqa: S301 -- deliberate wire format, see module docstring
    if tag == _TAG_SHM:
        return _loads_shm(body)
    msg = f"Unknown codec frame tag: {tag}"
    raise ValueError(msg)


def _dumps_shm(obj: Any) -> bytes:
    """Serialize ``obj``, diverting large out-of-band buffers into shared memory."""
    oob: list[pickle.PickleBuffer] = []

    def buffer_callback(buffer: pickle.PickleBuffer) -> bool:
        # Return False -> out-of-band (we handle it); True -> keep in-band.
        if buffer.raw().nbytes >= SHM_MIN_BYTES:
            oob.append(buffer)
            return False
        return True

    meta = pickle.dumps(obj, protocol=_PROTOCOL, buffer_callback=buffer_callback)

    segments: list[tuple[str, int]] = []
    for buffer in oob:
        raw = buffer.raw()
        nbytes = raw.nbytes
        shm = _create_segment(nbytes)
        shm.buf[:nbytes] = raw  # type: ignore[index]  # buf is non-None right after create
        shm.close()  # unmap here; the segment stays alive until the consumer unlinks it
        segments.append((shm.name, nbytes))

    envelope = {"meta": meta, "segments": segments}
    return bytes([_TAG_SHM]) + pickle.dumps(envelope, protocol=_PROTOCOL)


def _loads_shm(body: memoryview) -> Any:
    """Reconstruct an object whose large buffers live in shared-memory segments."""
    envelope = pickle.loads(body)  # noqa: S301 -- deliberate wire format, see module docstring
    buffers = [_consume_segment(name, nbytes) for name, nbytes in envelope["segments"]]
    return pickle.loads(envelope["meta"], buffers=buffers)  # noqa: S301 -- deliberate wire format, see module docstring


def _create_segment(nbytes: int) -> shared_memory.SharedMemory:
    """Create a segment the producer owns but does not track (the consumer unlinks it)."""
    if _SHM_HAS_TRACK:
        return shared_memory.SharedMemory(create=True, size=nbytes, track=False)  # type: ignore[call-arg]
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    _untrack(shm)
    return shm


def _consume_segment(name: str, nbytes: int) -> bytearray:
    """Open a segment, copy it into owned (writable) memory, then close and unlink it."""
    shm = (
        shared_memory.SharedMemory(name=name, track=False)  # type: ignore[call-arg]
        if _SHM_HAS_TRACK
        else shared_memory.SharedMemory(name=name)
    )
    try:
        return bytearray(shm.buf[:nbytes])  # type: ignore[index]  # buf is non-None right after attach
    finally:
        shm.close()
        with contextlib.suppress(FileNotFoundError):  # already gone; nothing to leak
            shm.unlink()
        if not _SHM_HAS_TRACK:
            # On 3.11/3.12 unlink() does not unregister; clear the tracker ourselves.
            _untrack(shm)


def _untrack(shm: shared_memory.SharedMemory) -> None:
    """Remove a segment from this process's resource_tracker (best effort)."""
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - tracker state is best-effort, never fatal
        LOGGER.debug(f"resource_tracker.unregister failed for {shm.name}: {exc!r}")
