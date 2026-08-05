"""Threaded node wiring `AsrMixin` speech-to-text into the pipeline."""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import AudioChunk, BaseMessage, TextChunk
from neudc.core.node.model.mixins.asr_mixin import AsrMixin
from neudc.nn.backends.asr.openai_compatible import OpenAICompatibleASRBackend


class AsrNode(AsrMixin, BaseThreadedNode):
    """A node that transcribes VAD-segmented speech into `TextChunk`s via a remote ASR service."""

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)

    def __init__(
        self,
        backend: Any,
        mailbox: Any,
        *,
        min_speech_duration: float = 0.3,
        id: str = "AsrNode",
    ) -> None:
        """Initialize the AsrNode.

        Args:
        ----
            backend: ASR backend used to transcribe buffered utterances.
            mailbox: Mailbox for sending `TextChunk` objects.
            min_speech_duration: Minimum buffered speech duration (seconds) worth transcribing.
            id: Identifier for this node instance.

        """
        AsrMixin.__init__(self, backend=backend, min_speech_duration=min_speech_duration)
        BaseThreadedNode.__init__(self, mailbox, id=id)

    @classmethod
    def from_config(cls: type[AsrNode], config: dict[str, Any]) -> AsrNode:
        """Create an AsrNode from a configuration dictionary.

        Args:
        ----
            config (dict): Configuration with keys `mailbox` and `asr_config` (a mapping
                with `model_id`, `base_url`, and optionally `api_key`/`timeout`), plus
                optionally `min_speech_duration`. `session_id`/`turn_id` are not
                configured here — they're stamped upstream by `VadNode`, which owns
                turn-boundary assignment (see `VadFilterMixin`), and propagated as-is.

        Returns:
        -------
            AsrNode: Instantiated AsrNode.

        """
        asr_config = config["asr_config"]
        backend = OpenAICompatibleASRBackend(
            model_id=asr_config["model_id"],
            base_url=asr_config["base_url"],
            api_key=asr_config.get("api_key"),
            timeout=asr_config.get("timeout", 60.0),
        )
        return AsrNode(
            backend=backend,
            mailbox=config["mailbox"],
            min_speech_duration=config.get("min_speech_duration", 0.3),
        )
