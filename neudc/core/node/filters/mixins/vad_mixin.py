"""Voice-Activity-Detection Filter Mixin."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from neudc.core.communication.messaging.types import ControlAction, ControlMessage
from neudc.core.node.filters.mixins.aec import REFERENCE_SOURCE, AdaptiveEchoCanceller, ReferenceBuffer
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

    Optional acoustic echo cancellation (AEC, `aec_enabled`): wire `AudioPlayerNode`'s
    `outputs` to this node so it also receives a copy of every block actually played
    (tagged `source=aec.REFERENCE_SOURCE`, see `AudioPlayerNode`). Such a chunk is
    routed straight into `ReferenceBuffer` and never treated as mic input or forwarded
    downstream. Ordinary mic chunks then get the estimated echo subtracted (via
    `AdaptiveEchoCanceller`) *before* the RMS/turn logic below runs, so both silence
    detection and whatever `AsrMixin`/ASR receives are working from the cleaned
    signal — real interruptions and real speech no longer need a separate, blunter
    energy threshold to tell apart from the assistant's own voice. Without AEC (the
    default), a speaker and mic close together (e.g. a laptop) will pick up the
    assistant's own TTS output as "the user talking again," repeatedly cancelling and
    restarting itself; the two mitigations below remain useful as an on-top safety net
    (or the whole fallback when AEC is off, e.g. headphones make it unnecessary):

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
        aec_enabled: Whether echo cancellation is active (see class docstring).

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
        aec_enabled: bool = False,
        aec_filter_length: int = 2048,
        aec_step_size: float = 0.3,
        mic_sample_rate: int = 16000,
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
            aec_enabled: Whether to subtract an estimated echo (from
                `AdaptiveEchoCanceller`) out of mic chunks before the RMS/turn logic
                runs. Requires `AudioPlayerNode`'s `outputs` wired back to this node
                (see class docstring) — with no reference audio arriving, this is a
                no-op regardless of the flag.
            aec_filter_length: Taps for the adaptive filter, at `mic_sample_rate`. See
                `AdaptiveEchoCanceller`.
            aec_step_size: NLMS adaptation rate (mu). See `AdaptiveEchoCanceller`.
            mic_sample_rate: Sample rate mic chunks arrive at — the rate reference
                audio is resampled to before cancellation (see `ReferenceBuffer`).

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

        self.aec_enabled = aec_enabled
        self._aec_filter_length = aec_filter_length
        self._canceller = AdaptiveEchoCanceller(filter_length=aec_filter_length, step_size=aec_step_size)
        self._reference_buffer = ReferenceBuffer(target_sample_rate=mic_sample_rate)

    def process(self, chunk: AudioChunk) -> AudioChunk | None:
        """Mark `chunk` as dropped if silent, and stamp/advance the turn it belongs to.

        A rising edge (silence -> speech) always starts a new turn. If a previous turn
        was still active, this *may* be a barge-in — the previous turn is cancelled
        over the control channel, but only if the onset clears
        `barge_in_energy_threshold` and `barge_in_cooldown_sec` has elapsed since the
        last cancel (see class docstring); otherwise the new turn still starts, it
        just doesn't interrupt the old one.

        A reference chunk (`chunk.source == aec.REFERENCE_SOURCE`, from
        `AudioPlayerNode`) is fed into the AEC reference buffer and consumed here —
        never treated as mic input, never forwarded downstream.

        Args:
        ----
            chunk: Incoming AudioChunk to inspect (mic input or AEC reference).

        Returns:
        -------
            The same AudioChunk, with `drop` set to True when it looks like silence,
            `samples` replaced by the echo-cancelled signal when AEC is active, and
            `session_id`/`turn_id` stamped for downstream turn tracking. `None` for a
            reference chunk (nothing to forward).

        """
        if chunk.source == REFERENCE_SOURCE:
            if self.aec_enabled:
                self._reference_buffer.push(chunk.samples, chunk.sample_rate)
            return None

        samples = chunk.samples
        if self.aec_enabled and samples.size > 0 and self._reference_buffer.is_active():
            window = self._reference_buffer.window(self._aec_filter_length, len(samples))
            if window is not None:
                samples = self._canceller.process(samples, window)
                chunk.samples = samples

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
