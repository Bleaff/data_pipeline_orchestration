"""Tests for OpenAICompatibleTTSBackend request/response handling, hermetic via a fake session."""

from __future__ import annotations

import wave
from io import BytesIO

import numpy as np
import pytest
import requests

from neudc.nn.backends.tts.openai_compatible import OpenAICompatibleTTSBackend, _decode_wav


def _make_wav_bytes(samples_int16: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples_int16.tobytes())
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content=b"", status_error=None):
        self.content = content
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def post(self, url, json, headers, timeout):
        self.last_call = {"url": url, "json": json, "headers": headers, "timeout": timeout}
        return self.response

    def close(self):
        pass


def test_requires_base_url():
    with pytest.raises(ValueError, match="requires a base_url"):
        OpenAICompatibleTTSBackend("some-model")


def test_headers_include_bearer_token_when_api_key_set():
    backend = OpenAICompatibleTTSBackend("m", base_url="http://fake.test/v1", api_key="secret")

    assert backend._headers()["Authorization"] == "Bearer secret"


def test_synthesize_sends_expected_payload_and_decodes_wav():
    backend = OpenAICompatibleTTSBackend("tts-1", voice="alloy", base_url="http://fake.test/v1")
    wav_bytes = _make_wav_bytes(np.array([0, 16384, -16384, 32767], dtype=np.int16), sample_rate=16000)
    backend._session = _FakeSession(_FakeResponse(content=wav_bytes))

    samples, sample_rate = backend.synthesize("hello")

    assert sample_rate == 16000
    assert len(samples) == 4
    assert samples.dtype == np.float32
    assert samples[0] == pytest.approx(0.0, abs=1e-6)
    assert samples[3] == pytest.approx(1.0, abs=1e-3)

    call = backend._session.last_call
    assert call["url"] == "http://fake.test/v1/audio/speech"
    assert call["json"]["model"] == "tts-1"
    assert call["json"]["voice"] == "alloy"
    assert call["json"]["input"] == "hello"
    assert call["json"]["response_format"] == "wav"


def test_synthesize_omits_voice_when_not_given():
    backend = OpenAICompatibleTTSBackend("tts-1", base_url="http://fake.test/v1")
    wav_bytes = _make_wav_bytes(np.zeros(10, dtype=np.int16), sample_rate=16000)
    backend._session = _FakeSession(_FakeResponse(content=wav_bytes))

    backend.synthesize("hi")

    assert "voice" not in backend._session.last_call["json"]


def test_synthesize_raises_on_http_error():
    backend = OpenAICompatibleTTSBackend("tts-1", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(status_error=requests.HTTPError("boom")))

    with pytest.raises(requests.HTTPError):
        backend.synthesize("hi")


def test_decode_wav_downmixes_stereo_to_mono():
    left_right = np.array([[100, 200], [300, 400]], dtype=np.int16)
    wav_bytes = _make_wav_bytes(left_right.reshape(-1), sample_rate=8000, channels=2)

    samples, sample_rate = _decode_wav(wav_bytes)

    assert sample_rate == 8000
    assert len(samples) == 2
