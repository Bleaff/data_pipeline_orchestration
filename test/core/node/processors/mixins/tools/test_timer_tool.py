"""Tests for TimerTool: immediate confirmation now, a reminder delivered later."""

from __future__ import annotations

from typing import ClassVar

import pytest

from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.core.node.processors.mixins.tools.timer_tool import MAX_TIMER_SECONDS, TimerTool


class _Collector:
    def __init__(self) -> None:
        self.sent: list = []

    def __call__(self, item) -> None:
        self.sent.append(item)


class _FakeTimer:
    """Stand-in for `threading.Timer`: records the schedule, never sleeps for real."""

    instances: ClassVar[list[_FakeTimer]] = []

    def __init__(self, seconds: float, callback) -> None:
        self.seconds = seconds
        self.callback = callback
        self.daemon = False
        self.started = False
        _FakeTimer.instances.append(self)

    def start(self) -> None:
        self.started = True

    def fire(self) -> None:
        self.callback()


@pytest.fixture(autouse=True)
def _reset_fake_timer():
    _FakeTimer.instances.clear()
    yield
    _FakeTimer.instances.clear()


def test_call_returns_immediate_confirmation_without_firing() -> None:
    delivered = _Collector()
    tool = TimerTool(delivered, scheduler=_FakeTimer)

    result = tool(seconds=300)

    assert "300" in result
    assert delivered.sent == []  # not fired yet


def test_call_includes_label_in_confirmation() -> None:
    delivered = _Collector()
    tool = TimerTool(delivered, scheduler=_FakeTimer)

    result = tool(seconds=60, label="pasta")

    assert "pasta" in result


def test_schedules_via_the_given_scheduler_as_a_daemon() -> None:
    delivered = _Collector()
    tool = TimerTool(delivered, scheduler=_FakeTimer)

    tool(seconds=5)

    assert len(_FakeTimer.instances) == 1
    scheduled = _FakeTimer.instances[0]
    assert scheduled.seconds == 5
    assert scheduled.started is True
    assert scheduled.daemon is True


def test_firing_delivers_a_reminder_text_chunk_with_the_session_id() -> None:
    delivered = _Collector()
    tool = TimerTool(delivered, session_id="sess-1", scheduler=_FakeTimer)

    tool(seconds=5, label="pasta")
    _FakeTimer.instances[0].fire()

    assert len(delivered.sent) == 1
    reminder = delivered.sent[0]
    assert reminder.source == "timer"
    assert reminder.session_id == "sess-1"
    assert "pasta" in reminder.text


def test_firing_without_a_label_uses_a_generic_message() -> None:
    delivered = _Collector()
    tool = TimerTool(delivered, scheduler=_FakeTimer)

    tool(seconds=5)
    _FakeTimer.instances[0].fire()

    assert delivered.sent[0].text == "Timer finished."


@pytest.mark.parametrize("seconds", [0, -5, MAX_TIMER_SECONDS + 1])
def test_rejects_seconds_out_of_range(seconds: float) -> None:
    tool = TimerTool(_Collector(), scheduler=_FakeTimer)

    with pytest.raises(ToolError, match="seconds must be"):
        tool(seconds=seconds)

    assert _FakeTimer.instances == []


def test_deliver_failure_is_logged_not_raised() -> None:
    def _broken(item) -> None:
        del item
        msg = "boom"
        raise RuntimeError(msg)

    tool = TimerTool(_broken, scheduler=_FakeTimer)
    tool(seconds=5)

    _FakeTimer.instances[0].fire()  # must not raise
