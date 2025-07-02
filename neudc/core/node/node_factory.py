"""Create node factory instances based on config.

This module provides a factory to create node instances from configuration
dictionaries. The factory knows how to create nodes of different types based on
the config.

"""

from __future__ import annotations

from typing import Any, ClassVar

from neudc.core.node.broadcast.image_saver import SaveImageNode
from neudc.core.node.processors.resize_node import ResizeNode
from neudc.core.node.processors.draw_node import DrawNode
from neudc.core.node.processors.resize_process_node import ResizeProcessNode
from neudc.core.node.readers.image_reader import FolderImageNode
from neudc.core.node.model.proc_det_inference import ProcessDetInference
from neudc.core.node.filters.hash_node import HashNode
class NodeFactory:
    """Factory to create node instances based on config."""

    NODE_CLASS_MAP: ClassVar = {
        "FolderImageNode": FolderImageNode,
        "SaveImageNode": SaveImageNode,
        "ResizeProcessNode": ResizeProcessNode,
        "ResizeNode": ResizeNode,
        "ProcessDetInference": ProcessDetInference,
        "DrawNode": DrawNode,
        "HashNode": HashNode,
    }

    @staticmethod
    def create(config: dict[str, Any], mailbox: Any, logger: Any) -> Any:
        """Create a node instance from its config.

        Args:
        ----
            config (dict): Node config.
            mailbox (Any): Precreated mailbox for this node.
            logger (Any): Logger for this node.

        Returns:
        -------
            Any: Instantiated node.

        """
        node_type = config["type"]
        node_class = NodeFactory.NODE_CLASS_MAP.get(node_type)

        if not node_class:
            msg = f"Unknown node type: {node_type}"
            raise ValueError(msg)

        config = dict(config)  # make a copy
        config["mailbox"] = mailbox
        config["logger"] = logger
        return node_class.from_config(config)
