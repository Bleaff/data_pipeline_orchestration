"""Dead-letter sink: a record of every message a node could not process.

Without this a failed message vanishes into a log line and there is no way to tell
afterwards *what* was lost. Each record is one JSON object on its own line, so the
file can be tailed while a pipeline runs and read back with any JSONL tooling.

One file per node (``<dir>/<node_id>.jsonl``): a node owns its file, and since a
process node is one process, appends never interleave with another writer.

Payloads (pixels, tensors) are **not** written by default — a dead-lettered frame
would otherwise cost megabytes on disk. Enable ``store_payload`` to also pickle the
message next to the record when reproducing the failure matters more than space.
"""

from __future__ import annotations

import json
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

from neudc.utils import LOGGER

# Enough of the traceback to identify the failure without unbounded records.
_MAX_TRACEBACK_CHARS = 4000

# Best-effort identity: fields worth recording if the message happens to carry them.
_IDENTITY_FIELDS = ("frame_id", "source_frame", "timestamp", "source", "session_id", "turn_id")


class DeadLetterSink:
    """Append-only JSONL record of messages that failed processing."""

    def __init__(self, directory: str | Path, store_payload: bool = False) -> None:
        """Initialize the sink.

        Args:
        ----
            directory: Where to write ``<node_id>.jsonl`` (created on first record).
            store_payload: Also pickle the message itself next to the record.

        """
        self.directory = Path(directory)
        self.store_payload = store_payload

    def record(self, node_id: str, item: Any, exc: BaseException | None) -> None:
        """Write one dead-letter record.

        Never raises: losing a diagnostic must not take down the node that was
        already having a bad time. Failures to write are logged instead.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            entry = self._build_entry(node_id, item, exc)
            if self.store_payload:
                entry["payload_file"] = self._store_payload(node_id, item, entry["timestamp"])
            path = self.directory / f"{node_id}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as write_error:
            LOGGER.error(f"[{node_id}] could not write dead-letter record to {self.directory}: {write_error!r}")

    def _build_entry(self, node_id: str, item: Any, exc: BaseException | None) -> dict[str, Any]:
        """Assemble the JSON record for a failed message."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "node_id": node_id,
            "message_type": type(item).__name__,
            "error_type": type(exc).__name__ if exc is not None else None,
            "error": str(exc) if exc is not None else None,
        }
        if exc is not None:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            entry["traceback"] = formatted[-_MAX_TRACEBACK_CHARS:]
        entry["message"] = _identity_of(item)
        return entry

    def _store_payload(self, node_id: str, item: Any, timestamp: float) -> str | None:
        """Pickle the message beside the record; return its path, or None if it could not be pickled."""
        payload_dir = self.directory / node_id
        payload_dir.mkdir(parents=True, exist_ok=True)
        path = payload_dir / f"{timestamp:.6f}.pkl"
        try:
            with path.open("wb") as handle:
                pickle.dump(item, handle, protocol=5)
        except (OSError, pickle.PicklingError, TypeError, AttributeError) as dump_error:
            # A payload that cannot be pickled is still worth a record without it.
            LOGGER.warning(f"[{node_id}] dead-letter payload not stored: {dump_error!r}")
            return None
        return str(path)


def _identity_of(item: Any) -> dict[str, Any]:
    """Pull whichever identifying fields the message happens to expose.

    Payload-agnostic on purpose: a frame has ``frame_id``/``source_frame``, other
    message types have none of them, and neither case is an error.
    """
    identity: dict[str, Any] = {}
    for field in _IDENTITY_FIELDS:
        value = item.get(field) if isinstance(item, dict) else getattr(item, field, None)
        if value is not None and isinstance(value, str | int | float | bool):
            identity[field] = value
    return identity
