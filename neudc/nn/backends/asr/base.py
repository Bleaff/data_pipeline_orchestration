"""ASR backend abstraction: the common interface pluggable speech-to-text providers implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ("BaseASRBackend",)


class BaseASRBackend(ABC):
    """Common interface every concrete ASR provider (self-hosted Whisper, ...) implements."""

    model_id: str

    @abstractmethod
    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Init of BaseASRBackend.

        Args:
        ----
            model_id (str): Provider-specific model identifier (e.g. `"whisper-large-v3"`).
            base_url (str | None): Provider endpoint, when it is not fixed by the provider itself.
            api_key (str | None): Credential for the provider, when required.
            timeout (float): Request timeout in seconds.

        """

    @abstractmethod
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Transcribe a chunk of PCM audio into text.

        Args:
        ----
            samples (np.ndarray): Mono PCM samples (float in [-1, 1] or int16).
            sample_rate (int): Sampling rate of `samples`, in Hz.

        Returns:
        -------
            str: The transcribed text.

        """
