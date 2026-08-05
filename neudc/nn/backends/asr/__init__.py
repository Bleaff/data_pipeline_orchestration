"""Pluggable ASR (speech-to-text) provider backends.

Mirrors `neudc.nn.backends.llm`'s `base.py` + concrete-class split, but without a
factory/enum dispatch layer: there is currently one concrete provider (a self-hosted
OpenAI-compatible transcription server), so an extra abstraction level has no second
implementation to justify it yet.
"""

from __future__ import annotations

from neudc.nn.backends.asr.base import BaseASRBackend
from neudc.nn.backends.asr.openai_compatible import OpenAICompatibleASRBackend

__all__ = (
    "BaseASRBackend",
    "OpenAICompatibleASRBackend",
)
