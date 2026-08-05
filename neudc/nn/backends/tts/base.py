"""TTS backend abstraction: the common interface pluggable text-to-speech providers implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ("BaseTTSBackend",)


class BaseTTSBackend(ABC):
    """Common interface every concrete TTS provider (self-hosted Piper/XTTS, ...) implements."""

    model_id: str

    @abstractmethod
    def __init__(
        self,
        model_id: str,
        *,
        voice: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Init of BaseTTSBackend.

        Args:
        ----
            model_id (str): Provider-specific model identifier.
            voice (str | None): Provider-specific voice/speaker name, when applicable.
            base_url (str | None): Provider endpoint, when it is not fixed by the provider itself.
            api_key (str | None): Credential for the provider, when required.
            timeout (float): Request timeout in seconds.

        """

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize `text` into PCM audio.

        Args:
        ----
            text (str): The text to speak.

        Returns:
        -------
            tuple[np.ndarray, int]: Mono float32 PCM samples in `[-1, 1]`, and their sample rate in Hz.

        """
