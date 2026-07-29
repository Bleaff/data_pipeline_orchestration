"""Factory responsible for creating and wiring mailboxes for nodes.

The factory is responsible for:
    1. Creating mailboxes for all nodes
    2. Wiring output connections between nodes

The factory takes the application configuration as a parameter.
The configuration must contain a list of node configurations.
Each node configuration must contain an "id" parameter.
The "outputs" parameter is optional and contains a list of target node_ids
that the node should send messages to.

Worker replicas (#15): a node config may set ``replicas: N`` to run N independent
instances of the same node type, each with its own mailbox. ``create_mailboxes``
always returns a ``list[ZMQMailbox]`` per node id (length 1 for the default,
non-replicated case) so every caller has one uniform shape to iterate. Wiring an
edge to a replicated target connects every producer replica's ``ZeroQueuePub`` to
*every* consumer replica's port, so ZMQ round-robins fairly across the whole group.
"""

from __future__ import annotations

from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox


class RoutingFactory:
    """Factory responsible for creating and wiring mailboxes for nodes."""

    def __init__(self, config: dict) -> None:
        """Store the pipeline configuration used to build and wire mailboxes."""
        self.config = config

    def create_mailboxes(self) -> dict[str, list[ZMQMailbox]]:
        """Create mailboxes for all nodes and wire their output queues.

        Returns
        -------
            dict[str, list[ZMQMailbox]]: Mapping of node_id to its list of ZMQMailbox
            replicas (one entry unless the node config sets ``replicas > 1``).

        """
        mailboxes: dict[str, list[ZMQMailbox]] = {}

        # 1. Create one or more mailboxes per node, depending on `replicas`.
        for node_cfg in self.config["nodes"]:
            node_id = node_cfg["id"]
            replicas = node_cfg.get("replicas", 1)
            if replicas <= 1:
                mailboxes[node_id] = [ZMQMailbox(name=node_id)]
            else:
                mailboxes[node_id] = [ZMQMailbox(name=f"{node_id}#{i}") for i in range(replicas)]

        # 2. Wire output connections: every producer replica connects to every port
        # of every consumer replica, so PUSH round-robins across the whole target group.
        for node_cfg in self.config["nodes"]:
            node_id = node_cfg["id"]
            outputs = node_cfg.get("outputs", [])
            for target_node_id in outputs:
                if target_node_id not in mailboxes:
                    msg = f"Target node '{target_node_id}' not found in mailboxes."
                    raise ValueError(msg)
                target_ports = [mb.consume_port for mb in mailboxes[target_node_id]]
                for producer_mailbox in mailboxes[node_id]:
                    producer_mailbox.add_publisher(target_node_id, target_ports)

        return mailboxes
