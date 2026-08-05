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
        session_id: str = "default",
        barge_in_energy_threshold: float | None = None,
        barge_in_cooldown_sec: float = 1.5,
        **kwargs: Any,
    ) -> None:
        """Initialize the VadNode with an RMS energy threshold."""
        VadFilterMixin.__init__(
            self,
            energy_threshold=energy_threshold,
            session_id=session_id,
            barge_in_energy_threshold=barge_in_energy_threshold,
            barge_in_cooldown_sec=barge_in_cooldown_sec,
        )
        BaseThreadedNode.__init__(self, *args, id=id, **kwargs)
        self.id = id

    @classmethod
    def from_config(cls: type[VadNode], config: dict[str, Any]) -> VadNode:
        """Create VadNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox', 'energy_threshold',
                and optionally 'session_id' (stamped onto every chunk/cancellation this
                node emits — owns turn-boundary assignment for barge-in, see
                `VadFilterMixin`), 'barge_in_energy_threshold' (defaults to
                `energy_threshold`; raise it above `energy_threshold` on a setup
                without echo cancellation, where the mic picks up the speaker), and
                'barge_in_cooldown_sec' (default 1.5 — minimum seconds between cancels,
                bounding a runaway self-triggering loop).

        Returns:
        -------
            VadNode: Instantiated VadNode.

        """
        return VadNode(
            mailbox=config["mailbox"],
            energy_threshold=config.get("energy_threshold", 0.01),
            session_id=config.get("session_id", "default"),
            barge_in_energy_threshold=config.get("barge_in_energy_threshold"),
            barge_in_cooldown_sec=config.get("barge_in_cooldown_sec", 1.5),
        )
