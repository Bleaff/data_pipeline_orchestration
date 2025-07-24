from __future__ import annotations

from typing import Any

from neudc.core.base.base_thread import BaseThreadedNode
from neudc.core.node.filters.mixins.hash_mixin import HashFilterMixin


class HashNode(HashFilterMixin, BaseThreadedNode):
    """A node that resizes incoming Frame objects to a target resolution."""

    def __init__(
        self,
        delta: int,
        hash_type: str,
        hash_size: int,
        id: str = "HashNode",
        *args: Any,
        **kwargs: Any,
    ) -> HashNode:
        """Initialize the HashNode with delta and type of hash."""
        HashFilterMixin.__init__(self, delta=delta, hash_type=hash_type, hash_size=hash_size)
        BaseThreadedNode.__init__(self, id=id, *args, **kwargs)
        self.id = id

    @classmethod
    def from_config(cls: type[HashNode], config: dict[str, Any]) -> HashNode:
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
            delta=config["delta"],
            hash_size=config["hash_size"],
            hash_type=config["hash_type"],
        )
