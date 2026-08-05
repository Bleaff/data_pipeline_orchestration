"""Pluggable TTS (text-to-speech) provider backends.

Mirrors `neudc.nn.backends.llm`'s `base.py` + concrete-class split, but without a
factory/enum dispatch layer: there is currently one concrete provider (a self-hosted
OpenAI-compatible speech server), so an extra abstraction level has no second
implementation to justify it yet.
"""

from __future__ import annotations

from neudc.nn.backends.tts.base import BaseTTSBackend
from neudc.nn.backends.tts.openai_compatible import OpenAICompatibleTTSBackend

__all__ = (
    "BaseTTSBackend",
    "OpenAICompatibleTTSBackend",
)
