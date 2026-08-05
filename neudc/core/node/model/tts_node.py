"""Threaded node wiring `TtsMixin` speech synthesis into the pipeline."""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import AudioChunk, BaseMessage, TextChunk
from neudc.core.node.model.mixins.tts_mixin import TtsMixin
from neudc.nn.backends.tts.openai_compatible import OpenAICompatibleTTSBackend


class TtsNode(TtsMixin, BaseThreadedNode):
    """A node that synthesizes LLM replies into `AudioChunk`s via a remote TTS service."""

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)

    def __init__(self, backend: Any, mailbox: Any, *, id: str = "TtsNode") -> None:
        """Initialize the TtsNode.

        Args:
        ----
            backend: TTS backend used to synthesize buffered text.
            mailbox: Mailbox for sending `AudioChunk` objects.
            id: Identifier for this node instance.

        """
        TtsMixin.__init__(self, backend=backend)
        BaseThreadedNode.__init__(self, mailbox, id=id)

    @classmethod
    def from_config(cls: type[TtsNode], config: dict[str, Any]) -> TtsNode:
        """Create a TtsNode from a configuration dictionary.

        Args:
        ----
            config (dict): Configuration with keys `mailbox` and `tts_config` (a mapping
                with `model_id`, `base_url`, and optionally `voice`/`api_key`/`timeout`).

        Returns:
        -------
            TtsNode: Instantiated TtsNode.

        """
        tts_config = config["tts_config"]
        backend = OpenAICompatibleTTSBackend(
            model_id=tts_config["model_id"],
            voice=tts_config.get("voice"),
            base_url=tts_config["base_url"],
            api_key=tts_config.get("api_key"),
            timeout=tts_config.get("timeout", 60.0),
        )
        return TtsNode(backend=backend, mailbox=config["mailbox"])
