"""HTTP client for any self-hosted ASR service speaking the OpenAI transcriptions wire format."""

from __future__ import annotations

import io
import wave

import numpy as np
import requests

from neudc.nn.backends.asr.base import BaseASRBackend

__all__ = ("OpenAICompatibleASRBackend",)

_TRANSCRIPTIONS_PATH = "/audio/transcriptions"


def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono PCM `samples` as 16-bit WAV bytes using the stdlib `wave` module.

    Accepts float samples in `[-1, 1]` (the convention used by `AudioChunk`/`sounddevice`)
    or samples already in `int16`; either way the wire format is 16-bit PCM WAV, which
    every OpenAI-compatible transcription server accepts without further negotiation.
    """
    if np.issubdtype(samples.dtype, np.floating):
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        pcm16 = samples.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class OpenAICompatibleASRBackend(BaseASRBackend):
    """Talks to any `/audio/transcriptions` endpoint following the OpenAI wire format.

    Self-hosted Whisper servers (e.g. `faster-whisper-server`) expose this same
    protocol, mirroring `OpenAICompatibleBackend` in `neudc.nn.backends.llm` for the
    chat-completions path.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenAICompatibleASRBackend class.

        Args:
        ----
            model_id (str): Provider-specific model identifier (e.g. `"Systran/faster-whisper-large-v3"`).
            base_url (str | None): Root URL of the OpenAI-compatible API (no trailing slash needed).
            api_key (str | None): Bearer token sent as `Authorization`, when the provider requires one.
            timeout (float): Request timeout in seconds.

        """
        if not base_url:
            msg = f"{type(self).__name__} requires a base_url"
            raise ValueError(msg)

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Transcribe a chunk of PCM audio into text via a synchronous HTTP call.

        Args:
        ----
            samples (np.ndarray): Mono PCM samples (float in [-1, 1] or int16).
            sample_rate (int): Sampling rate of `samples`, in Hz.

        Returns:
        -------
            str: The transcribed text.

        """
        wav_bytes = _encode_wav(samples, sample_rate)
        response = self._session.post(
            f"{self.base_url}{_TRANSCRIPTIONS_PATH}",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": self.model_id},
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["text"]

    def __del__(self) -> None:
        """Release the underlying HTTP session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
