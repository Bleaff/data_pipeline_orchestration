"""HTTP client for any self-hosted TTS service speaking the OpenAI speech wire format."""

from __future__ import annotations

import io
import wave

import numpy as np
import requests

from neudc.nn.backends.tts.base import BaseTTSBackend

__all__ = ("OpenAICompatibleTTSBackend",)

_SPEECH_PATH = "/audio/speech"


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes into mono float32 samples in `[-1, 1]` and their sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())

    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm16 = pcm16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    samples = (pcm16.astype(np.float32)) / 32768.0
    return samples, sample_rate


class OpenAICompatibleTTSBackend(BaseTTSBackend):
    """Talks to any `/audio/speech` endpoint following the OpenAI wire format.

    Self-hosted TTS wrappers (e.g. `openedai-speech` in front of Piper/XTTS) expose
    this same protocol, mirroring `OpenAICompatibleBackend` in `neudc.nn.backends.llm`
    for the chat-completions path.
    """

    def __init__(
        self,
        model_id: str,
        *,
        voice: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenAICompatibleTTSBackend class.

        Args:
        ----
            model_id (str): Provider-specific model identifier (e.g. `"tts-1"`).
            voice (str | None): Provider-specific voice/speaker name (e.g. `"alloy"`).
            base_url (str | None): Root URL of the OpenAI-compatible API (no trailing slash needed).
            api_key (str | None): Bearer token sent as `Authorization`, when the provider requires one.
            timeout (float): Request timeout in seconds.

        """
        if not base_url:
            msg = f"{type(self).__name__} requires a base_url"
            raise ValueError(msg)

        self.model_id = model_id
        self.voice = voice
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize `text` into PCM audio via a synchronous HTTP call.

        Args:
        ----
            text (str): The text to speak.

        Returns:
        -------
            tuple[np.ndarray, int]: Mono float32 PCM samples in `[-1, 1]`, and their sample rate in Hz.

        """
        payload = {"model": self.model_id, "input": text, "response_format": "wav"}
        if self.voice:
            payload["voice"] = self.voice
        response = self._session.post(
            f"{self.base_url}{_SPEECH_PATH}",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _decode_wav(response.content)

    def __del__(self) -> None:
        """Release the underlying HTTP session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
