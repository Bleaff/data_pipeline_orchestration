"""Tests for VadFilterMixin's RMS-energy voice-activity heuristic and barge-in."""

from __future__ import annotations

import time

import numpy as np

from neudc.core.communication.messaging.types import AudioChunk, ControlAction
from neudc.core.node.filters.mixins.vad_mixin import VadFilterMixin


class _FakeMailbox:
    """Records every control message sent, without any real transport."""

    def __init__(self) -> None:
        self.sent_control = []

    def send_control(self, message) -> None:
        self.sent_control.append(message)


def _chunk(samples: np.ndarray) -> AudioChunk:
    return AudioChunk(timestamp=1.0, source="mic", samples=samples, sample_rate=16000, channels=1)


def _loud(n: int = 1600) -> np.ndarray:
    t = np.arange(n) / 16000
    return (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(n: int = 1600) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _tone(amplitude: float, n: int = 1600) -> np.ndarray:
    t = np.arange(n) / 16000
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_loud_chunk_passes_through() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)

    result = mixin.process(_chunk(_loud()))

    assert result.drop is False


def test_silent_chunk_is_dropped() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)

    result = mixin.process(_chunk(_silence()))

    assert result.drop is True


def test_empty_samples_do_not_crash() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)
    empty = np.array([], dtype=np.float32)

    result = mixin.process(_chunk(empty))

    assert result.drop is True


def test_every_chunk_is_stamped_with_session_id_and_current_turn() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05, session_id="sess-1")

    result = mixin.process(_chunk(_loud()))

    assert result.session_id == "sess-1"
    assert result.turn_id == 1


def test_first_ever_utterance_sends_no_cancel() -> None:
    # Nothing was in flight yet, so onset for turn 1 has nothing to interrupt.
    mixin = VadFilterMixin(energy_threshold=0.05)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))

    assert mixin.mailbox.sent_control == []


def test_continuous_speech_does_not_advance_the_turn_or_cancel() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05)
    mixin.mailbox = _FakeMailbox()

    first = mixin.process(_chunk(_loud()))
    second = mixin.process(_chunk(_loud()))

    assert first.turn_id == second.turn_id == 1
    assert mixin.mailbox.sent_control == []


def test_onset_after_the_previous_utterance_still_cancels_it_even_if_its_reply_already_finished() -> None:
    # VAD can't tell "turn 1's reply already finished playing" from "still in
    # flight" — it only knows speech resumed, so every onset after turn 1 fires a
    # cancel for it regardless. That's fine: cancelling an already-completed turn is
    # a harmless no-op downstream (BaseNode._is_cancelled just returns True for a
    # turn nothing is processing anymore).
    mixin = VadFilterMixin(energy_threshold=0.05)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))  # turn 1 speech
    mixin.process(_chunk(_silence()))  # turn 1 ends
    second = mixin.process(_chunk(_loud()))  # turn 2 onset

    assert second.turn_id == 2
    assert len(mixin.mailbox.sent_control) == 1
    assert mixin.mailbox.sent_control[0].turn_id == 1


def test_speech_onset_mid_reply_cancels_the_previous_turn() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05, session_id="sess-1")
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))  # turn 1 onset
    mixin.process(_chunk(_silence()))  # turn 1 utterance ends, reply presumably starts
    second = mixin.process(_chunk(_loud()))  # user barges in on turn 1's reply

    assert second.turn_id == 2
    assert len(mixin.mailbox.sent_control) == 1
    cancelled = mixin.mailbox.sent_control[0]
    assert cancelled.action is ControlAction.CANCEL
    assert cancelled.session_id == "sess-1"
    assert cancelled.turn_id == 1


# === Echo mitigation: barge_in_energy_threshold / barge_in_cooldown_sec. Neither is
# real acoustic echo cancellation (see VadFilterMixin's docstring) — these just bound
# how easily a mic-picks-up-the-speaker false trigger can fire/cascade.


def test_onset_below_barge_in_threshold_starts_a_new_turn_without_cancelling() -> None:
    # RMS of a 0.1-amplitude tone is well above energy_threshold (so it's "speech",
    # a new turn starts) but below barge_in_energy_threshold (so it's not trusted
    # enough to interrupt) — the shape of a quiet echo bleed-through.
    mixin = VadFilterMixin(energy_threshold=0.02, barge_in_energy_threshold=0.3)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))  # turn 1
    mixin.process(_chunk(_silence()))  # turn 1 ends
    second = mixin.process(_chunk(_tone(0.1)))  # quiet onset

    assert second.turn_id == 2  # still segmented as a new turn
    assert mixin.mailbox.sent_control == []  # but too quiet to trust as a real barge-in


def test_onset_above_barge_in_threshold_still_cancels() -> None:
    mixin = VadFilterMixin(energy_threshold=0.02, barge_in_energy_threshold=0.3)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))  # RMS ~0.57, clears 0.3
    mixin.process(_chunk(_silence()))
    mixin.process(_chunk(_loud()))

    assert len(mixin.mailbox.sent_control) == 1


def test_cooldown_suppresses_a_second_cancel_right_after_the_first() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05, barge_in_cooldown_sec=60.0)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))  # turn 1
    mixin.process(_chunk(_silence()))
    mixin.process(_chunk(_loud()))  # turn 2 onset, cancels turn 1
    mixin.process(_chunk(_silence()))
    mixin.process(_chunk(_loud()))  # turn 3 onset, immediately after — cooldown blocks it

    assert len(mixin.mailbox.sent_control) == 1
    assert mixin.mailbox.sent_control[0].turn_id == 1


def test_cancel_still_advances_the_turn_even_when_cooldown_blocks_it() -> None:
    # Barge-in being suppressed must not stop ASR from segmenting the new utterance
    # under its own (new) turn id — only the *cancel* is gated, not turn tracking.
    mixin = VadFilterMixin(energy_threshold=0.05, barge_in_cooldown_sec=60.0)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))
    mixin.process(_chunk(_silence()))
    mixin.process(_chunk(_loud()))  # cancels turn 1
    mixin.process(_chunk(_silence()))
    third = mixin.process(_chunk(_loud()))  # cooldown blocks the cancel...

    assert third.turn_id == 3  # ...but this is still correctly turn 3, not stuck on 2


def test_cooldown_expiring_allows_the_next_cancel() -> None:
    mixin = VadFilterMixin(energy_threshold=0.05, barge_in_cooldown_sec=0.05)
    mixin.mailbox = _FakeMailbox()

    mixin.process(_chunk(_loud()))
    mixin.process(_chunk(_silence()))
    mixin.process(_chunk(_loud()))  # cancels turn 1
    mixin.process(_chunk(_silence()))
    time.sleep(0.1)
    mixin.process(_chunk(_loud()))  # cooldown has elapsed, cancels turn 2

    assert [c.turn_id for c in mixin.mailbox.sent_control] == [1, 2]
