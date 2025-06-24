from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.filters.mixins.hash_mixin import HashFilterMixin


class HashNode(HashFilterMixin, BaseThreadedNode):
    """A node that resizes incoming Frame objects to a target resolution."""

    def __init__(self, delta: int, hash_type: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the HashNode with delta and type of hash"""
        self.delta: int = delta
        self.type_of_hash: str = hash_type
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: dict[str, Any]) -> HashNode:
        """Create HashNode from configuration dictionary.

        Args:
        ----
            config (dict): Configuration dict containing 'mailbox', 'logger', 'delta', and 'hash_type'.

        Returns:
        -------
            HashNode: Instantiated ResizeNode.

        """
        return HashNode(
            mailbox=config["mailbox"],
            logger=config["logger"],
            delta=config["delta"],
            hash_type=config["hash_type"],
        )
