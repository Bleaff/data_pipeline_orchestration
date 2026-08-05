"""Threaded node wiring `LlmMixin` chat completion into the pipeline."""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import BaseMessage, TextChunk
from neudc.core.node.model.mixins.llm_mixin import LlmMixin
from neudc.nn.backends.llm.config import LLMBackendConfig, build_llm_backend


class LlmNode(LlmMixin, BaseThreadedNode):
    """A node that answers ASR transcriptions via a pluggable LLM backend, keeping conversation history."""

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = (TextChunk,)

    def __init__(
        self,
        backend: Any,
        mailbox: Any,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        id: str = "LlmNode",
    ) -> None:
        """Initialize the LlmNode.

        Args:
        ----
            backend: LLM backend used to generate replies.
            mailbox: Mailbox for sending `TextChunk` objects.
            system_prompt: Optional system message seeding the conversation history.
            stream: Whether to stream content deltas instead of one final reply.
            temperature: Sampling temperature forwarded to the backend.
            max_tokens: Upper bound on generated tokens, provider default if `None`.
            id: Identifier for this node instance.

        """
        LlmMixin.__init__(
            self,
            backend=backend,
            system_prompt=system_prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        BaseThreadedNode.__init__(self, mailbox, id=id)

    @classmethod
    def from_config(cls: type[LlmNode], config: dict[str, Any]) -> LlmNode:
        """Create an LlmNode from a configuration dictionary.

        Args:
        ----
            config (dict): Configuration with keys `mailbox` and `llm_config` (validated
                as `LLMBackendConfig`: `provider`, `model_id`, `base_url`, `stream`,
                `temperature`, `max_tokens`, `timeout`), plus optionally `system_prompt`.

        Returns:
        -------
            LlmNode: Instantiated LlmNode.

        """
        llm_backend_config = LLMBackendConfig(**config["llm_config"])
        backend = build_llm_backend(llm_backend_config)
        return LlmNode(
            backend=backend,
            mailbox=config["mailbox"],
            system_prompt=config.get("system_prompt"),
            stream=llm_backend_config.stream,
            temperature=llm_backend_config.temperature,
            max_tokens=llm_backend_config.max_tokens,
        )
