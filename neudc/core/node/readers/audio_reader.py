"""AudioReaderNode class.

This node reads live audio from a pluggable device (microphone / stream) and emits
:class:`~neudc.core.communication.messaging.types.AudioChunk` messages. It is the
first *unbounded* live source in the repo: unlike ``FolderImageNode``, which reads a
finite set of files and paces itself with an explicit ``time.sleep``, this node has
no natural end and no fixed rate of its own — the device's blocking ``read`` call
*is* the pacing. There is deliberately no ``time.sleep`` anywhere in ``process()``.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import AudioChunk, BaseMessage
from neudc.utils import LOGGER


@runtime_checkable
class AudioDevice(Protocol):
    """Structural interface for a pluggable audio input backend.

    ``read`` is the pacing primitive for :class:`AudioReaderNode`: a real
    implementation blocks until ``n_frames`` samples are available (a microphone
    or network stream), the same way a real file descriptor paces a blocking
    read. Returning fewer samples than requested (e.g. an empty array) or raising
    signals that the device is exhausted/closed.
    """

    def read(self, n_frames: int) -> np.ndarray:
        """Block until ``n_frames`` samples are available and return them."""
        ...

    def close(self) -> None:
        """Release any resources held by the device."""
        ...


class SoundDeviceAudioDevice:
    """Real microphone/stream backend built on the optional ``sounddevice`` package.

    ``sounddevice`` is not a hard dependency of this package (see the ``audio``
    extra in ``pyproject.toml``); it is imported lazily here, inside ``__init__``,
    so importing :mod:`neudc.core.node.readers.audio_reader` — and therefore
    building any other pipeline through :class:`~neudc.core.node.node_factory.NodeFactory`
    — never requires it to be installed. Construct this explicitly and pass it as
    the ``device`` for a real-hardware :class:`AudioReaderNode`; ``from_config``
    does not build one implicitly (see its docstring).
    """

    def __init__(self, sample_rate: int, channels: int = 1, device: int | str | None = None) -> None:
        """Open an input stream on the given (or default) hardware device.

        Args:
        ----
            sample_rate: Sampling rate to open the stream at, in Hz.
            channels: Number of input channels to capture.
            device: ``sounddevice`` device index/name, or ``None`` for the system default.

        Raises:
        ------
            ImportError: If the optional ``sounddevice`` package is not installed.

        """
        try:
            import sounddevice as sd
        except ImportError as e:
            msg = "SoundDeviceAudioDevice requires the optional 'sounddevice' package: pip install -e '.[audio]'"
            raise ImportError(msg) from e
        self._stream = sd.InputStream(samplerate=sample_rate, channels=channels, device=device)
        self._stream.start()

    def read(self, n_frames: int) -> np.ndarray:
        """Block until ``n_frames`` samples are captured and return them."""
        data, overflowed = self._stream.read(n_frames)
        if overflowed:
            LOGGER.warning("Audio input overflowed; some samples may have been dropped.")
        return data

    def close(self) -> None:
        """Stop and close the underlying input stream."""
        self._stream.stop()
        self._stream.close()


class AudioReaderNode(BaseThreadedNode):
    """A node that reads live audio from a pluggable device and emits `AudioChunk` messages.

    Unlike `FolderImageNode` (a finite, self-paced source), this is an unbounded
    stream: `process()` never calls `self.stop()` on its own success path. The
    blocking `device.read()` call is what sets the pace; the node only stops
    itself when the device signals it is exhausted or errors out.
    """

    emits: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)

    def __init__(
        self,
        device: AudioDevice,
        mailbox: Any,
        sample_rate: int,
        channels: int = 1,
        chunk_duration: float = 0.5,
        source: str = "mic",
    ) -> None:
        """Initialize AudioReaderNode.

        Args:
        ----
            device (AudioDevice): Pluggable audio backend; its blocking `read` paces the stream.
            mailbox (Any): Mailbox for sending `AudioChunk` objects.
            sample_rate (int): Sampling rate of the audio stream, in Hz.
            channels (int): Number of audio channels.
            chunk_duration (float): Duration of each emitted chunk, in seconds.
            source (str): Value stamped onto `AudioChunk.source` (device/stream name).

        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.source = source
        super().__init__(mailbox)

    @staticmethod
    def from_config(config: dict[str, Any]) -> AudioReaderNode:
        """Create an AudioReaderNode from a configuration dictionary.

        Real hardware access needs a live Python object (an `AudioDevice`), which
        cannot come from plain YAML the way `folder_path` or `sample_rate` can. So
        this requires an explicit `device` already present in `config` — built by
        whatever assembles the pipeline (the same way `mailbox` itself is injected
        by `NodeFactory.create` rather than read from YAML). For real hardware, pass
        a `SoundDeviceAudioDevice` instance (requires the optional `audio` extra).

        Args:
        ----
            config (dict): Configuration with keys `device`, `mailbox`, `sample_rate`,
                and optionally `channels`, `chunk_duration`, `source`.

        Returns:
        -------
            AudioReaderNode: Instantiated AudioReaderNode.

        Raises:
        ------
            ValueError: If `config` has no `device`.

        """
        device = config.get("device")
        if device is None:
            msg = (
                "AudioReaderNode.from_config requires an explicit 'device' (an AudioDevice) in "
                "config; there is no default hardware backend wired in from YAML. Construct one "
                "(e.g. SoundDeviceAudioDevice, which needs the optional 'audio' extra) and inject "
                "it into the config dict before building the pipeline."
            )
            raise ValueError(msg)
        return AudioReaderNode(
            device=device,
            mailbox=config["mailbox"],
            sample_rate=config["sample_rate"],
            channels=config.get("channels", 1),
            chunk_duration=config.get("chunk_duration", 0.5),
            source=config.get("source", "mic"),
        )

    def _collect_data(self) -> bool:
        return True

    def process(self, *args: Any, **kwargs: Any) -> AudioChunk | None:
        """Block on the device for one chunk's worth of samples and wrap it into an `AudioChunk`.

        Returns
        -------
            AudioChunk or None: The next chunk, or None if the device errored out or
            reported end of stream (in which case the reader also stops itself).

        """
        del args, kwargs

        n_frames = int(self.sample_rate * self.chunk_duration)
        try:
            samples = self.device.read(n_frames)
        except Exception as e:  # noqa: BLE001 -- device backends raise arbitrary, backend-specific errors.
            LOGGER.exception("Audio device read failed; stopping reader.", exc_info=e)
            self.stop()
            return None

        if samples is None or len(samples) == 0:
            LOGGER.info("Audio device reported end of stream; stopping reader.")
            self.stop()
            return None

        return AudioChunk(
            timestamp=time.time(),
            source=self.source,
            samples=samples,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
