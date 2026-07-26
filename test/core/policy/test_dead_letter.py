"""Tests for the dead-letter sink (Stage 9, #11).

The sink is the answer to "what did we lose and why". It must record enough to
identify the message, must not blow up disk with pixel payloads by default, and must
never itself take down a node that is already failing.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from neudc.core.communication.messaging.types import Frame
from neudc.core.policy import DeadLetterSink


def _frame(frame_id: int = 7) -> Frame:
    return Frame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        timestamp=1.5,
        source_frame="images/src.png",
        frame_id=frame_id,
        boxes=[],
        frame_id_last=10,
        drop=False,
    )


def _read_records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_captures_error_and_message_identity(tmp_path) -> None:
    sink = DeadLetterSink(tmp_path)
    try:
        # Raise-and-catch (rather than a helper) so the captured traceback is a real one.
        raise ValueError("model exploded")  # noqa: TRY301, TRY003, EM101
    except ValueError as error:
        sink.record("detector", _frame(), error)

    records = _read_records(tmp_path / "detector.jsonl")
    assert len(records) == 1
    entry = records[0]
    assert entry["node_id"] == "detector"
    assert entry["error_type"] == "ValueError"
    assert entry["error"] == "model exploded"
    assert "ValueError: model exploded" in entry["traceback"]
    # Enough to find the offending input again.
    assert entry["message_type"] == "Frame"
    assert entry["message"]["frame_id"] == 7
    assert entry["message"]["source_frame"] == "images/src.png"


def test_records_append_one_line_per_failure(tmp_path) -> None:
    sink = DeadLetterSink(tmp_path)
    error = RuntimeError("nope")

    sink.record("detector", _frame(1), error)
    sink.record("detector", _frame(2), error)

    records = _read_records(tmp_path / "detector.jsonl")
    assert [entry["message"]["frame_id"] for entry in records] == [1, 2]


def test_each_node_writes_its_own_file(tmp_path) -> None:
    # One process per node, one file per node: appends never interleave.
    sink = DeadLetterSink(tmp_path)
    sink.record("detector", _frame(), RuntimeError("a"))
    sink.record("resizer", _frame(), RuntimeError("b"))

    assert sorted(path.name for path in tmp_path.glob("*.jsonl")) == ["detector.jsonl", "resizer.jsonl"]


def test_payload_is_not_stored_by_default(tmp_path) -> None:
    # A dead-lettered frame would otherwise cost megabytes per record.
    DeadLetterSink(tmp_path).record("detector", _frame(), RuntimeError("x"))

    entry = _read_records(tmp_path / "detector.jsonl")[0]
    assert "payload_file" not in entry
    assert list(tmp_path.glob("**/*.pkl")) == []


def test_store_payload_writes_a_replayable_pickle(tmp_path) -> None:
    DeadLetterSink(tmp_path, store_payload=True).record("detector", _frame(3), RuntimeError("x"))

    entry = _read_records(tmp_path / "detector.jsonl")[0]
    with Path(entry["payload_file"]).open("rb") as handle:
        restored = pickle.load(handle)  # noqa: S301 -- loading a pickle this same test just wrote

    assert restored.frame_id == 3
    assert np.array_equal(restored.image, np.zeros((2, 2, 3), dtype=np.uint8))


def test_unpicklable_payload_still_produces_a_record(tmp_path) -> None:
    sink = DeadLetterSink(tmp_path, store_payload=True)

    sink.record("detector", lambda: None, RuntimeError("x"))  # a lambda cannot be pickled

    entry = _read_records(tmp_path / "detector.jsonl")[0]
    assert entry["payload_file"] is None
    assert entry["error"] == "x"


def test_message_without_identity_fields_is_still_recorded(tmp_path) -> None:
    # Non-CV payloads have no frame_id/source_frame, and that is not an error.
    DeadLetterSink(tmp_path).record("node", {"anything": 1}, RuntimeError("x"))

    entry = _read_records(tmp_path / "node.jsonl")[0]
    assert entry["message"] == {}
    assert entry["message_type"] == "dict"


def test_unwritable_directory_does_not_raise(tmp_path) -> None:
    # Losing a diagnostic must not take down the node that was already failing.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    DeadLetterSink(blocker / "dead").record("node", _frame(), RuntimeError("x"))  # must not raise
