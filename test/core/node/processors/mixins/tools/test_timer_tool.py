"""Tests for TimerTool: immediate confirmation now, a reminder delivered later."""

from __future__ import annotations

from typing import ClassVar

import pytest

from neudc.core.node.processors.mixins.tools.base import ToolError
from neudc.core.node.processors.mixins.tools.timer_tool import MAX_TIMER_SECONDS, TimerTool


class _FakeMailbox:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, item) -> None:
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
    mailbox = _FakeMailbox()
    tool = TimerTool(mailbox=mailbox, scheduler=_FakeTimer)

    result = tool(seconds=300)

    assert "300" in result
    assert mailbox.sent == []  # not fired yet


def test_call_includes_label_in_confirmation() -> None:
    mailbox = _FakeMailbox()
    tool = TimerTool(mailbox=mailbox, scheduler=_FakeTimer)

    result = tool(seconds=60, label="pasta")

    assert "pasta" in result


def test_schedules_via_the_given_scheduler_as_a_daemon() -> None:
    mailbox = _FakeMailbox()
    tool = TimerTool(mailbox=mailbox, scheduler=_FakeTimer)

    tool(seconds=5)

    assert len(_FakeTimer.instances) == 1
    scheduled = _FakeTimer.instances[0]
    assert scheduled.seconds == 5
    assert scheduled.started is True
    assert scheduled.daemon is True


def test_firing_sends_a_reminder_text_chunk_with_the_session_id() -> None:
    mailbox = _FakeMailbox()
    tool = TimerTool(mailbox=mailbox, session_id="sess-1", scheduler=_FakeTimer)

    tool(seconds=5, label="pasta")
    _FakeTimer.instances[0].fire()

    assert len(mailbox.sent) == 1
    reminder = mailbox.sent[0]
    assert reminder.source == "timer"
    assert reminder.session_id == "sess-1"
    assert "pasta" in reminder.text


def test_firing_without_a_label_uses_a_generic_message() -> None:
    mailbox = _FakeMailbox()
    tool = TimerTool(mailbox=mailbox, scheduler=_FakeTimer)

    tool(seconds=5)
    _FakeTimer.instances[0].fire()

    assert mailbox.sent[0].text == "Timer finished."


@pytest.mark.parametrize("seconds", [0, -5, MAX_TIMER_SECONDS + 1])
def test_rejects_seconds_out_of_range(seconds: float) -> None:
    tool = TimerTool(mailbox=_FakeMailbox(), scheduler=_FakeTimer)

    with pytest.raises(ToolError, match="seconds must be"):
        tool(seconds=seconds)

    assert _FakeTimer.instances == []


def test_mailbox_send_failure_is_logged_not_raised() -> None:
    class _BrokenMailbox:
        def send(self, item) -> None:
            del item
            msg = "boom"
            raise RuntimeError(msg)

    tool = TimerTool(mailbox=_BrokenMailbox(), scheduler=_FakeTimer)
    tool(seconds=5)

    _FakeTimer.instances[0].fire()  # must not raise
