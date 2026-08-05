"""Tests for OpenAICompatibleASRBackend request/response handling, hermetic via a fake session."""

from __future__ import annotations

import wave
from io import BytesIO

import numpy as np
import pytest
import requests

from neudc.nn.backends.asr.openai_compatible import OpenAICompatibleASRBackend, _encode_wav


def _wav_frame_count_and_rate(wav_bytes: bytes) -> tuple[int, int]:
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.getnframes(), wav_file.getframerate()


class _FakeResponse:
    def __init__(self, json_data=None, status_error=None):
        self._json = json_data
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def post(self, url, files, data, headers, timeout):
        self.last_call = {"url": url, "files": files, "data": data, "headers": headers, "timeout": timeout}
        return self.response

    def close(self):
        pass


def test_requires_base_url():
    with pytest.raises(ValueError, match="requires a base_url"):
        OpenAICompatibleASRBackend("some-model")


def test_headers_include_bearer_token_when_api_key_set():
    backend = OpenAICompatibleASRBackend("m", base_url="http://fake.test/v1", api_key="secret")

    assert backend._headers()["Authorization"] == "Bearer secret"


def test_headers_omit_auth_when_no_api_key():
    backend = OpenAICompatibleASRBackend("m", base_url="http://fake.test/v1")

    assert "Authorization" not in backend._headers()


def test_transcribe_returns_text_and_sends_wav_multipart():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(json_data={"text": "hello world"}))
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(1600) / 16000)).astype(np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == "hello world"
    call = backend._session.last_call
    assert call["url"] == "http://fake.test/v1/audio/transcriptions"
    assert call["data"]["model"] == "test-model"
    filename, wav_bytes, content_type = call["files"]["file"]
    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    assert _wav_frame_count_and_rate(wav_bytes) == (1600, 16000)


def test_transcribe_raises_on_http_error():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(status_error=requests.HTTPError("boom")))
    samples = np.zeros(1600, dtype=np.float32)

    with pytest.raises(requests.HTTPError):
        backend.transcribe(samples, sample_rate=16000)


def test_encode_wav_round_trips_sample_count_and_rate():
    samples = np.linspace(-1.0, 1.0, 800, dtype=np.float32)

    wav_bytes = _encode_wav(samples, sample_rate=8000)

    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() == 800
        assert wav_file.getframerate() == 8000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
