"""AudioPlayerNode class.

This node plays incoming :class:`~neudc.core.communication.messaging.types.AudioChunk`
messages out through a pluggable output device (speaker / stream). It is the output-side
mirror of :class:`~neudc.core.node.readers.audio_reader.AudioReaderNode`: real hardware
access needs a live `AudioOutputDevice` object, which plain YAML cannot express, so
`from_config` requires one already present in `config` (see its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import AudioChunk, BaseMessage
from neudc.utils import LOGGER

if TYPE_CHECKING:
    import numpy as np

#: How often (seconds of audio) playback pauses to re-check whether this chunk's turn
#: was cancelled — the granularity of barge-in interruption. Small enough to feel
#: instant, large enough not to fragment `device.write()` into pointless tiny calls.
DEFAULT_PLAYBACK_CHECK_INTERVAL_SEC = 0.1


@runtime_checkable
class AudioOutputDevice(Protocol):
    """Structural interface for a pluggable audio output backend."""

    def write(self, samples: np.ndarray) -> None:
        """Block until `samples` have been played (or queued for playback)."""
        ...

    def close(self) -> None:
        """Release any resources held by the device."""
        ...


class SoundDeviceAudioOutputDevice:
    """Real speaker/stream backend built on the optional `sounddevice` package.

    Mirrors `SoundDeviceAudioDevice` in `audio_reader.py`: `sounddevice` is imported
    lazily here, inside `__init__`, so importing this module — and therefore building
    any other pipeline through `NodeFactory` — never requires it to be installed.
    """

    def __init__(self, sample_rate: int, channels: int = 1, device: int | str | None = None) -> None:
        """Open an output stream on the given (or default) hardware device.

        Args:
        ----
            sample_rate: Sampling rate to open the stream at, in Hz.
            channels: Number of output channels.
            device: `sounddevice` device index/name, or `None` for the system default.

        Raises:
        ------
            ImportError: If the optional `sounddevice` package is not installed.

        """
        try:
            import sounddevice as sd
        except ImportError as e:
            msg = "SoundDeviceAudioOutputDevice requires the optional 'sounddevice' package: pip install -e '.[audio]'"
            raise ImportError(msg) from e
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=channels, device=device)
        self._stream.start()

    def write(self, samples: np.ndarray) -> None:
        """Block until `samples` have been written to the output stream."""
        self._stream.write(samples)

    def close(self) -> None:
        """Stop and close the underlying output stream."""
        self._stream.stop()
        self._stream.close()


class AudioPlayerNode(BaseThreadedNode):
    """A node that plays incoming `AudioChunk` messages through a pluggable output device.

    A sink: it consumes `AudioChunk`s and emits nothing, mirroring `SaveImageNode` for
    the CV path. A chunk marked `drop` is skipped rather than played.

    Plays each chunk in small blocks rather than one `device.write()` call, re-checking
    cancellation (`BaseNode._is_cancelled`) between blocks — this is what makes
    barge-in (see `VadFilterMixin`) actually cut off audio already playing, not just
    skip chunks that hadn't started yet: the framework's generic per-message
    cancellation check in `BaseNode._handle` runs once, *before* `process()` starts,
    so on its own it can't interrupt a single call already in progress against one
    (potentially many-seconds-long) reply.
    """

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = ()

    def __init__(
        self,
        device: AudioOutputDevice,
        mailbox: Any,
        *,
        playback_check_interval: float = DEFAULT_PLAYBACK_CHECK_INTERVAL_SEC,
        id: str = "AudioPlayerNode",
    ) -> None:
        """Initialize AudioPlayerNode.

        Args:
        ----
            device (AudioOutputDevice): Pluggable audio output backend.
            mailbox (Any): Mailbox for receiving `AudioChunk` objects.
            playback_check_interval (float): Seconds of audio played per block between
                cancellation checks — the granularity of barge-in interruption.
            id (str): Identifier for this node instance.

        """
        self.device = device
        self.playback_check_interval = playback_check_interval
        super().__init__(mailbox, id=id)

    @staticmethod
    def from_config(config: dict[str, Any]) -> AudioPlayerNode:
        """Create an AudioPlayerNode from a configuration dictionary.

        Real hardware access needs a live Python object (an `AudioOutputDevice`),
        which cannot come from plain YAML the way other parameters can — same
        reasoning as `AudioReaderNode.from_config`. For real hardware, pass a
        `SoundDeviceAudioOutputDevice` instance (requires the optional `audio` extra).

        Args:
        ----
            config (dict): Configuration with keys `device` and `mailbox`.

        Returns:
        -------
            AudioPlayerNode: Instantiated AudioPlayerNode.

        Raises:
        ------
            ValueError: If `config` has no `device`.

        """
        device = config.get("device")
        if device is None:
            msg = (
                "AudioPlayerNode.from_config requires an explicit 'device' (an AudioOutputDevice) "
                "in config; there is no default hardware backend wired in from YAML. Construct one "
                "(e.g. SoundDeviceAudioOutputDevice, which needs the optional 'audio' extra) and "
                "inject it into the config dict before building the pipeline."
            )
            raise ValueError(msg)
        return AudioPlayerNode(
            device=device,
            mailbox=config["mailbox"],
            playback_check_interval=config.get("playback_check_interval", DEFAULT_PLAYBACK_CHECK_INTERVAL_SEC),
        )

    def process(self, chunk: AudioChunk) -> None:
        """Play `chunk` through the output device in blocks, unless it is marked `drop`.

        Stops early — without playing the rest — if `chunk`'s turn gets cancelled
        partway through (barge-in): checked between blocks of
        `playback_check_interval` seconds each, not just once up front.

        Args:
        ----
            chunk: Incoming AudioChunk to play.

        """
        if chunk.drop:
            return

        block_frames = max(1, int(chunk.sample_rate * self.playback_check_interval))
        samples = chunk.samples
        for start in range(0, len(samples), block_frames):
            if self._is_cancelled(chunk.session_id, chunk.turn_id):
                LOGGER.info(f"[{self.id}] playback of turn {chunk.turn_id} interrupted (barge-in)")
                return
            self.device.write(samples[start : start + block_frames])
