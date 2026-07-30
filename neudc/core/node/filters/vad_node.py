"""Threaded node wiring `VadFilterMixin` energy-based voice-activity detection into the pipeline."""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.communication.messaging.types import AudioChunk, BaseMessage
from neudc.core.node.filters.mixins.vad_mixin import VadFilterMixin


class VadNode(VadFilterMixin, BaseThreadedNode):
    """A node that marks incoming `AudioChunk`s that look like silence as droppable."""

    accepts: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)
    emits: ClassVar[tuple[type[BaseMessage], ...]] = (AudioChunk,)

    def __init__(
        self,
        energy_threshold: float,
        id: str = "VadNode",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the VadNode with an RMS energy threshold."""
        VadFilterMixin.__init__(self, energy_threshold=energy_threshold)
        BaseThreadedNode.__init__(self, *args, id=id, **kwargs)
        self.id = id

    @classmethod
    def from_config(cls: type[VadNode], config: dict[str, Any]) -> VadNode:
        """Create VadNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox' and 'energy_threshold'.

        Returns:
        -------
            VadNode: Instantiated VadNode.

        """
        return VadNode(
            mailbox=config["mailbox"],
            energy_threshold=config.get("energy_threshold", 0.01),
        )
