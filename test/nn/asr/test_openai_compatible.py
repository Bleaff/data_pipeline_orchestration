"""Tests for OpenAICompatibleASRBackend request/response handling, hermetic via a fake session."""

from __future__ import annotations

import wave
from io import BytesIO

import numpy as np
import pytest
import requests

from neudc.nn.backends.asr.openai_compatible import OpenAICompatibleASRBackend, _encode_wav, _looks_like_silence

_REAL_SPEECH_SEGMENT = {"no_speech_prob": 0.0004}
_SILENCE_SEGMENT = {"no_speech_prob": 0.89}


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
    backend._session = _FakeSession(
        _FakeResponse(json_data={"text": "hello world", "segments": [_REAL_SPEECH_SEGMENT]}),
    )
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(1600) / 16000)).astype(np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == "hello world"
    call = backend._session.last_call
    assert call["url"] == "http://fake.test/v1/audio/transcriptions"
    assert call["data"]["model"] == "test-model"
    assert call["data"]["response_format"] == "verbose_json"  # default no_speech_threshold requires it
    filename, wav_bytes, content_type = call["files"]["file"]
    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    assert _wav_frame_count_and_rate(wav_bytes) == (1600, 16000)


def test_transcribe_sends_language_when_given():
    backend = OpenAICompatibleASRBackend("test-model", language="ru", base_url="http://fake.test/v1")
    backend._session = _FakeSession(
        _FakeResponse(json_data={"text": "привет", "segments": [_REAL_SPEECH_SEGMENT]}),
    )
    samples = np.zeros(1600, dtype=np.float32)

    backend.transcribe(samples, sample_rate=16000)

    assert backend._session.last_call["data"]["language"] == "ru"


def test_transcribe_omits_language_when_not_given():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(
        _FakeResponse(json_data={"text": "hello", "segments": [_REAL_SPEECH_SEGMENT]}),
    )
    samples = np.zeros(1600, dtype=np.float32)

    backend.transcribe(samples, sample_rate=16000)

    assert "language" not in backend._session.last_call["data"]


def test_transcribe_raises_on_http_error():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(status_error=requests.HTTPError("boom")))
    samples = np.zeros(1600, dtype=np.float32)

    with pytest.raises(requests.HTTPError):
        backend.transcribe(samples, sample_rate=16000)


# === no_speech_threshold: filtering out Whisper hallucinating text on silence/noise
# (verified against a real deployed faster-whisper-server — see _looks_like_silence).


def test_transcribe_suppresses_hallucinated_text_on_high_no_speech_prob():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(
        _FakeResponse(json_data={"text": "Продолжение следует...", "segments": [_SILENCE_SEGMENT]}),
    )
    samples = np.zeros(1600, dtype=np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == ""


def test_transcribe_keeps_text_on_low_no_speech_prob():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(
        _FakeResponse(json_data={"text": "hello world", "segments": [_REAL_SPEECH_SEGMENT]}),
    )
    samples = np.zeros(1600, dtype=np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == "hello world"


def test_transcribe_treats_no_segments_as_silence():
    backend = OpenAICompatibleASRBackend("test-model", base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(json_data={"text": "something", "segments": []}))
    samples = np.zeros(1600, dtype=np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == ""


def test_no_speech_threshold_none_disables_the_check_and_verbose_json_request():
    backend = OpenAICompatibleASRBackend("test-model", no_speech_threshold=None, base_url="http://fake.test/v1")
    backend._session = _FakeSession(_FakeResponse(json_data={"text": "Продолжение следует..."}))
    samples = np.zeros(1600, dtype=np.float32)

    out = backend.transcribe(samples, sample_rate=16000)

    assert out == "Продолжение следует..."
    assert "response_format" not in backend._session.last_call["data"]


def test_looks_like_silence_averages_across_multiple_segments():
    payload = {"segments": [{"no_speech_prob": 0.9}, {"no_speech_prob": 0.1}]}  # avg 0.5

    assert _looks_like_silence(payload, threshold=0.6) is False
    assert _looks_like_silence(payload, threshold=0.4) is True


def test_looks_like_silence_missing_no_speech_prob_defaults_to_zero():
    payload = {"segments": [{}]}

    assert _looks_like_silence(payload, threshold=0.6) is False


def test_encode_wav_round_trips_sample_count_and_rate():
    samples = np.linspace(-1.0, 1.0, 800, dtype=np.float32)

    wav_bytes = _encode_wav(samples, sample_rate=8000)

    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() == 800
        assert wav_file.getframerate() == 8000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1
