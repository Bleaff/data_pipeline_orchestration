"""Create node instances from configuration dictionaries.

The factory maps a node ``type`` string to the class that implements it and builds
an instance via that class's ``from_config``. Node classes are imported lazily, on
first use, so building a lightweight (e.g. CPU-only) pipeline does not require the
optional dependencies of unrelated nodes (torch, ultralytics, imagehash, ...).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

from neudc.core.policy import ErrorPolicy

if TYPE_CHECKING:
    from neudc.core.communication.messaging.types import BaseMessage


class NodeFactory:
    """Factory to create node instances based on config."""

    # node type -> (module path, class name). Imported on demand in `_resolve`.
    NODE_IMPORTS: ClassVar[dict[str, tuple[str, str]]] = {
        "FolderImageNode": ("neudc.core.node.readers.image_reader", "FolderImageNode"),
        "SaveImageNode": ("neudc.core.node.broadcast.image_saver", "SaveImageNode"),
        "ResizeProcessNode": ("neudc.core.node.processors.resize_process_node", "ResizeProcessNode"),
        "ResizeNode": ("neudc.core.node.processors.resize_node", "ResizeNode"),
        "ProcessDetInference": ("neudc.core.node.model.proc_det_inference", "ProcessDetInference"),
        "DrawNode": ("neudc.core.node.processors.draw_node", "DrawNode"),
        "HashNode": ("neudc.core.node.filters.hash_node", "HashNode"),
        "ProcessBlurInference": ("neudc.core.node.model.proc_blur_inference", "ProcessBlurInference"),
        "ProcessEmbeddingInference": ("neudc.core.node.model.proc_embedding_inference", "ProcessEmbeddingInference"),
        "ProcessDetBatchInference": ("neudc.core.node.model.proc_det_batch_inference", "ProcessDetBatchInference"),
        "ActiveLearning": ("neudc.core.node.model.active_learning_node", "ActiveLearning"),
        "CreateDataset": ("neudc.core.node.processors.create_dataset_node", "CreateDataset"),
        "TextNormalizeNode": ("neudc.core.node.processors.text_normalize_node", "TextNormalizeNode"),
    }

    @staticmethod
    def _resolve(node_type: str) -> Any:
        """Import and return the node class for ``node_type``."""
        try:
            module_path, class_name = NodeFactory.NODE_IMPORTS[node_type]
        except KeyError:
            msg = f"Unknown node type: {node_type}"
            raise ValueError(msg) from None
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def get_accepts(node_type: str) -> tuple[type[BaseMessage], ...]:
        """Return the payload types ``node_type`` declares it can consume (#35)."""
        return NodeFactory._resolve(node_type).accepts

    @staticmethod
    def get_emits(node_type: str) -> tuple[type[BaseMessage], ...]:
        """Return the payload types ``node_type`` declares it can produce (#35)."""
        return NodeFactory._resolve(node_type).emits

    @staticmethod
    def create(config: dict[str, Any], mailbox: Any) -> Any:
        """Create a node instance from its config.

        Args:
        ----
            config (dict): Node config. Must contain a ``type`` key.
            mailbox (Any): Precreated mailbox for this node.

        Returns:
        -------
            Any: Instantiated node.

        """
        node_type = config["type"]
        node_class = NodeFactory._resolve(node_type)

        config = config.copy()  # do not mutate the caller's dict
        config["mailbox"] = mailbox
        config.pop("type", None)
        config.pop("outputs", None)  # routing-only key, not a node argument
        config.pop("replicas", None)  # routing-only key (#15), consumed by RoutingFactory
        config.pop("autoscale", None)  # routing-only key (#15), consumed by ReplicaAutoscaler
        # Error handling is uniform across node types, so it is wired here instead of
        # in every from_config (see neudc.core.policy).
        error_policy = config.pop("error_policy", None)
        node_id = config.get("id") or node_type

        node = node_class.from_config(config)
        node.id = node_id
        node.error_policy = ErrorPolicy.from_config(error_policy, node_id=node_id)
        return node
