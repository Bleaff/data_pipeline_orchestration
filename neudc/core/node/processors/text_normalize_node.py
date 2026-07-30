"""Threaded node wiring `TextNormalizeMixin` text normalization into the pipeline."""

from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import TextChunk
from neudc.core.node.processors.mixins.text_normalize_mixin import TextNormalizeMixin


class TextNormalizeNode(TextNormalizeMixin, BaseThreadedNode):
    """A node that strips/lowercases the text of incoming `TextChunk` objects.

    A genuinely non-CV node (`TextChunk -> TextChunk`), used as a real fixture for the
    payload-compat validation added in #35/#36, alongside `Frame`-only CV nodes.
    """

    #: Non-CV payload contract (#35): overrides `BaseNode`'s `(Frame,)` default.
    accepts = (TextChunk,)
    emits = (TextChunk,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TextNormalizeNode, forwarding all arguments through the MRO."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: dict[str, Any]) -> TextNormalizeNode:
        """Create TextNormalizeNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox'.

        Returns:
        -------
            TextNormalizeNode: Instantiated TextNormalizeNode.

        """
        return TextNormalizeNode(
            mailbox=config["mailbox"],
        )
