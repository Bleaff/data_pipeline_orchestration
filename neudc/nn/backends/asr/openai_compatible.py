"""HTTP client for any self-hosted ASR service speaking the OpenAI transcriptions wire format."""

from __future__ import annotations

import io
import wave

import numpy as np

from neudc.nn.backends.asr.base import BaseASRBackend
from neudc.nn.backends.http_session import build_retrying_session

__all__ = ("OpenAICompatibleASRBackend",)

_TRANSCRIPTIONS_PATH = "/audio/transcriptions"


def _looks_like_silence(payload: dict, threshold: float) -> bool:
    """Whether a `verbose_json` transcription looks like a hallucination on silence/noise.

    Whisper (and derivatives like faster-whisper) are well documented to hallucinate
    plausible-sounding filler text — "thanks for watching", "to be continued", stock
    acknowledgement phrases — when fed near-silent or noisy audio instead of real
    speech, rather than returning nothing. Confirmed directly against a real deployed
    faster-whisper-server: low-amplitude noise (the shape of residual echo/AEC
    leftover that still clears a VAD energy threshold) transcribed to a fluent,
    unrelated sentence with `no_speech_prob` above 0.89, while genuine speech scored
    below 0.001. `no_speech_prob` — only present in the `verbose_json` response
    format, per segment — is the model's own confidence that a segment has no real
    speech in it; averaged across segments and compared against `threshold`. No
    segments at all (nothing detected) counts as silence too.
    """
    segments = payload.get("segments") or []
    if not segments:
        return True
    avg_no_speech_prob = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
    return avg_no_speech_prob > threshold


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
        language: str | None = None,
        no_speech_threshold: float | None = 0.6,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the OpenAICompatibleASRBackend class.

        Args:
        ----
            model_id (str): Provider-specific model identifier (e.g. `"Systran/faster-whisper-large-v3"`).
            language (str | None): ISO-639-1 language hint (e.g. `"ru"`) forwarded to
                the provider. `None` leaves language auto-detection on — Whisper does
                this per request from the audio itself, which works but can misfire on
                short utterances or in a multilingual setup; pinning it removes that
                ambiguity when the assistant is only ever used in one language.
            no_speech_threshold (float | None): Reject a transcription (return `""`
                instead) when its average `no_speech_prob` (from a `verbose_json`
                response) exceeds this — filters out Whisper hallucinating filler
                text on silence/noise rather than transcribing real speech (see
                `_looks_like_silence`). `None` disables the check entirely and
                requests the provider's normal (`json`) response format instead, for
                a provider whose `/audio/transcriptions` doesn't support
                `verbose_json`/`no_speech_prob`.
            base_url (str | None): Root URL of the OpenAI-compatible API (no trailing slash needed).
            api_key (str | None): Bearer token sent as `Authorization`, when the provider requires one.
            timeout (float): Request timeout in seconds.

        """
        if not base_url:
            msg = f"{type(self).__name__} requires a base_url"
            raise ValueError(msg)

        self.model_id = model_id
        self.language = language
        self.no_speech_threshold = no_speech_threshold
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = build_retrying_session()

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
            str: The transcribed text, or `""` if `no_speech_threshold` is set and
            the audio looks like silence/noise Whisper hallucinated text from.

        """
        wav_bytes = _encode_wav(samples, sample_rate)
        data = {"model": self.model_id}
        if self.language:
            data["language"] = self.language
        if self.no_speech_threshold is not None:
            data["response_format"] = "verbose_json"
        response = self._session.post(
            f"{self.base_url}{_TRANSCRIPTIONS_PATH}",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=data,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if self.no_speech_threshold is not None and _looks_like_silence(payload, self.no_speech_threshold):
            return ""
        return payload["text"]

    def __del__(self) -> None:
        """Release the underlying HTTP session."""
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
