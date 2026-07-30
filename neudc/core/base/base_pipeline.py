"""Base class for pipeline.

This module contains the base class for a pipeline. A pipeline is a special type of node that
can be used to run other nodes in separate processes, allowing for parallel execution of node graphs.

The base pipeline class provides a basic process node that can be used to run a list of nodes
in a separate process. It also supports a task configuration that can be used to pass additional
configuration to the pipeline.

"""

from __future__ import annotations

import logging
import time
from multiprocessing import Process
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neudc.core.base.base_node import BaseNode, EventLike


class BasePipeline(Process):
    """Base class for a pipeline.

    This class extends the Process class to allow for parallel execution of node graphs.
    """

    def __init__(
        self,
        name: str,
        stop_event: EventLike,
        nodes_config: list[dict[str, Any]],
        task_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the pipeline process.

        Args:
        ----
            name (str): Name of the pipeline, also used as the logger name.
            stop_event (EventLike): Event signaling the pipeline to shut down.
            nodes_config (list[dict[str, Any]]): Per-node configuration entries for the pipeline.
            task_config (dict[str, Any] | None): Task-level configuration shared across nodes.

        """
        super().__init__()
        self.name = name
        self.nodes_config = nodes_config
        self.task_config = task_config or {}
        self.stop_event = stop_event
        self.nodes: list[BaseNode] = []

    def run(self) -> None:
        """Run the pipeline process.

        Initializes logging, sets up the routing, creates the nodes, and starts them.
        Also keeps the process alive until interrupted.
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(self.name)

        # 1. Routing
        from neudc.core.communication.messaging.routing_factory import RoutingFactory
        from neudc.core.node.node_factory import NodeFactory

        router = RoutingFactory({"nodes": self.nodes_config})
        mailbox_map = router.create_mailboxes()

        # 2. Node init. Each node id maps to a list of mailboxes: length 1 unless the
        # node config sets `replicas > 1` (#15), in which case one node instance is
        # built per replica mailbox and given a distinct, suffixed id so per-node
        # metrics/health/dead-letter files do not collide across replicas.
        for node_config in self.nodes_config:
            node_id = node_config["id"]
            replica_mailboxes = mailbox_map[node_id]
            for i, mailbox in enumerate(replica_mailboxes):
                node = NodeFactory.create(node_config, mailbox=mailbox)
                if len(replica_mailboxes) > 1:
                    node.id = f"{node_id}#{i}"
                self.nodes.append(node)

        # Start all nodes
        logger.info(f"Starting pipeline '{self.name}' with {len(self.nodes)} nodes.")
        for node in self.nodes:
            node.start()

        # 3. Keep alive, watching for a node that gave up under an `on_error: fail`
        # policy. Deliberately not `node.status()`: health also goes false when a node
        # merely sits idle, and an idle pipeline must not shut itself down.
        try:
            while not self.stop_event.is_set():
                broken = [node.id for node in self.nodes if node.failed()]
                if broken:
                    logger.critical(f"Node(s) {broken} failed under their error policy; stopping pipeline.")
                    break
                time.sleep(0.1)  # avoid busy-spinning the CPU
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by keyboard.")
        finally:
            logger.info("Stopping pipeline...")
            for node in self.nodes:
                node.stop()
            logger.info("Pipeline stopped.")
