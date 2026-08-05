"""Voice-Activity-Detection Filter Mixin."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from neudc.core.communication.messaging.types import ControlAction, ControlMessage
from neudc.utils import LOGGER

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import AudioChunk


class VadFilterMixin:
    """Mixin for marking silent `AudioChunk`s as droppable, by RMS energy.

    A simple, dependency-free voice-activity heuristic: compute the RMS energy of
    the chunk's samples and compare it against `energy_threshold`. No ML model is
    needed for this — matches the level of the existing `HashFilterMixin` (a plain
    signal-level heuristic, not an inference node).

    Also owns turn-boundary assignment for the whole downstream chain: a rising edge
    (silence -> speech) starts a new turn, stamped as `session_id`/`turn_id` on every
    chunk from here on. This is what makes barge-in possible — VAD is the only node
    that can know "the user started talking again" *before* that speech has even been
    transcribed, so it is the only node early enough to cancel a reply already in
    flight. On a rising edge with a turn already active, it broadcasts
    `ControlMessage(action=CANCEL)` for that previous turn over the control channel
    (wired via this node's `control_outputs` in YAML, e.g. to `llm`/`tts`/`player`) —
    interrupting whatever downstream node is still working on it.

    Without acoustic echo cancellation (not implemented here — this is a plain
    energy heuristic, it has no reference signal of what the speaker is currently
    playing to subtract), a speaker and mic close together (e.g. a laptop) will pick
    up the assistant's own TTS output as "the user talking again," repeatedly
    cancelling and restarting itself. Two mitigations, both tunable, neither a real
    fix — the actual fix is either physical (headphones, or a mic/speaker placed far
    apart) or a proper AEC stage upstream of this node:

    - `barge_in_energy_threshold` (default: same as `energy_threshold`, i.e. no extra
      gating): a *second*, normally-higher bar an onset's RMS must clear to actually
      fire a cancel. Ordinary utterance segmentation (is this chunk speech or
      silence, for `AsrMixin`'s buffering) still only uses `energy_threshold` — this
      only gates whether an onset is trusted enough to interrupt a reply in flight.
    - `barge_in_cooldown_sec`: minimum time between two cancels. Without this, a
      sustained echo re-triggers onset detection every time playback resumes after
      each self-cancellation, cancelling again immediately — an unbounded loop (the
      "talking to itself" failure mode). This turns that into at most one spurious
      interruption per cooldown window instead of a cascade.

    Attributes
    ----------
        energy_threshold: RMS energy below which a chunk is considered silence.
        session_id: Stamped onto every chunk `process()` returns; also carried on the
            `ControlMessage` a barge-in sends, so cancellation lands on the exact
            (session_id, turn_id) pair a downstream node is checking against.
        barge_in_energy_threshold: RMS energy an onset must clear to fire a cancel.
        barge_in_cooldown_sec: Minimum seconds between two cancels this mixin sends.

    """

    #: Provided by whichever node class this is mixed into (e.g. `BaseThreadedNode`
    #: sets it in `BaseNode.__init__`) — declared here only so `_cancel_turn` type-checks;
    #: never assigned by this mixin itself.
    mailbox: Any

    def __init__(
        self,
        energy_threshold: float = 0.01,
        *,
        session_id: str = "default",
        barge_in_energy_threshold: float | None = None,
        barge_in_cooldown_sec: float = 1.5,
    ) -> None:
        """Initialize the mixin.

        Args:
        ----
            energy_threshold: RMS energy below which a chunk is marked as dropped.
            session_id: Stamped onto every chunk and every cancellation this node emits.
            barge_in_energy_threshold: RMS energy an onset must clear to fire a cancel
                (as opposed to just starting a new turn without interrupting the old
                one). Defaults to `energy_threshold` (no extra gating); raise it above
                `energy_threshold` on a setup without echo cancellation, where the mic
                picks up the speaker (see class docstring).
            barge_in_cooldown_sec: Minimum seconds between two cancels this mixin
                sends — bounds a runaway self-triggering loop to at most one spurious
                interruption per window, rather than a cascade.

        """
        self.energy_threshold = energy_threshold
        self.session_id = session_id
        self.barge_in_energy_threshold = (
            energy_threshold if barge_in_energy_threshold is None else barge_in_energy_threshold
        )
        self.barge_in_cooldown_sec = barge_in_cooldown_sec
        self._turn_id = 0
        self._in_speech = False
        self._last_cancel_at: float | None = None

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Mark `chunk` as dropped if silent, and stamp/advance the turn it belongs to.

        A rising edge (silence -> speech) always starts a new turn. If a previous turn
        was still active, this *may* be a barge-in — the previous turn is cancelled
        over the control channel, but only if the onset clears
        `barge_in_energy_threshold` and `barge_in_cooldown_sec` has elapsed since the
        last cancel (see class docstring); otherwise the new turn still starts, it
        just doesn't interrupt the old one.

        Args:
        ----
            chunk: Incoming AudioChunk to inspect.

        Returns:
        -------
            The same AudioChunk, with `drop` set to True when it looks like silence,
            and `session_id`/`turn_id` stamped for downstream turn tracking.

        """
        samples = chunk.samples
        rms = self._rms(samples) if samples.size > 0 else 0.0
        is_speech = samples.size > 0 and rms >= self.energy_threshold

        if is_speech and not self._in_speech:
            previous_turn_id = self._turn_id
            self._turn_id += 1
            if previous_turn_id > 0 and rms >= self.barge_in_energy_threshold and self._cooled_down():
                self._cancel_turn(previous_turn_id)
                self._last_cancel_at = time.time()
        self._in_speech = is_speech

        chunk.drop = not is_speech
        chunk.session_id = self.session_id
        chunk.turn_id = self._turn_id
        return chunk

    def _cooled_down(self) -> bool:
        return self._last_cancel_at is None or (time.time() - self._last_cancel_at) >= self.barge_in_cooldown_sec

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))

    def _cancel_turn(self, turn_id: int) -> None:
        """Broadcast a CANCEL for `turn_id` over the control channel (barge-in)."""
        LOGGER.info(f"Speech detected mid-reply — cancelling turn {turn_id} (barge-in)")
        self.mailbox.send_control(
            ControlMessage(
                timestamp=time.time(),
                session_id=self.session_id,
                turn_id=turn_id,
                action=ControlAction.CANCEL,
            ),
        )
