"""Factory responsible for creating and wiring mailboxes for nodes.

The factory is responsible for:
    1. Creating mailboxes for all nodes
    2. Wiring output connections between nodes

The factory takes the application configuration as a parameter.
The configuration must contain a list of node configurations.
Each node configuration must contain an "id" parameter.
The "outputs" parameter is optional and contains a list of target node_ids
that the node should send messages to.
The "control_outputs" parameter is optional and contains a list of target node_ids
that the node should send priority control-plane messages to (e.g. turn
cancellation / barge-in, #41) — wired into the target's control channel, entirely
separate from "outputs"/the data FIFO.

Worker replicas (#15): a node config may set ``replicas: N`` to run N independent
instances of the same node type, each with its own mailbox. ``create_mailboxes``
always returns a ``list[ZMQMailbox]`` per node id (length 1 for the default,
non-replicated case) so every caller has one uniform shape to iterate. Wiring a data
edge to a replicated target connects every producer replica's ``ZeroQueuePub`` to
*every* consumer replica's port, so ZMQ round-robins fairly across the whole group.
Wiring a control edge instead broadcasts to every consumer replica's control port:
a cancellation can't be round-robined, since we don't know in advance which replica
is processing the affected turn.
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
        mailboxes = self._create_replica_mailboxes()
        self._wire_data_outputs(mailboxes)
        self._wire_control_outputs(mailboxes)
        return mailboxes

    def _create_replica_mailboxes(self) -> dict[str, list[ZMQMailbox]]:
        """Create one or more mailboxes per node, depending on `replicas` (#15).

        `message_queue_size`/`queue_policy` (#38) are per-node config, defaulting to
        the historical `20`/`"block"` so an untouched config behaves exactly as before.
        """
        mailboxes: dict[str, list[ZMQMailbox]] = {}
        for node_cfg in self.config["nodes"]:
            node_id = node_cfg["id"]
            replicas = node_cfg.get("replicas", 1)
            message_queue_size = node_cfg.get("message_queue_size", 20)
            queue_policy = node_cfg.get("queue_policy", "block")
            if replicas <= 1:
                mailboxes[node_id] = [
                    ZMQMailbox(name=node_id, message_queue_size=message_queue_size, queue_policy=queue_policy)
                ]
            else:
                mailboxes[node_id] = [
                    ZMQMailbox(
                        name=f"{node_id}#{i}",
                        message_queue_size=message_queue_size,
                        queue_policy=queue_policy,
                    )
                    for i in range(replicas)
                ]
        return mailboxes

    def _wire_data_outputs(self, mailboxes: dict[str, list[ZMQMailbox]]) -> None:
        """Wire "outputs": every producer replica connects to every consumer replica's
        port, so PUSH round-robins fairly across the whole target group (#15).
        """  # noqa: D205
        for node_cfg in self.config["nodes"]:
            node_id = node_cfg["id"]
            for target_node_id in node_cfg.get("outputs", []):
                target_mailboxes = self._require_target(mailboxes, target_node_id)
                target_ports = [mb.consume_port for mb in target_mailboxes]
                for producer_mailbox in mailboxes[node_id]:
                    producer_mailbox.add_publisher(target_node_id, target_ports)

    def _wire_control_outputs(self, mailboxes: dict[str, list[ZMQMailbox]]) -> None:
        """Wire "control_outputs" (#41), independent of the data edges above.

        A node with no "control_outputs" simply never wires its control mailbox to
        anything, so an untouched config behaves exactly as before. Every producer
        replica broadcasts to every consumer replica's control port (#15): a
        cancellation can't be round-robined, since we don't know in advance which
        replica is processing the affected turn.
        """
        for node_cfg in self.config["nodes"]:
            node_id = node_cfg["id"]
            for target_node_id in node_cfg.get("control_outputs", []):
                target_mailboxes = self._require_target(mailboxes, target_node_id)
                target_control_ports = [mb.control_consume_port for mb in target_mailboxes]
                for producer_mailbox in mailboxes[node_id]:
                    producer_mailbox.add_control_publisher(target_node_id, target_control_ports)

    @staticmethod
    def _require_target(mailboxes: dict[str, list[ZMQMailbox]], target_node_id: str) -> list[ZMQMailbox]:
        """Look up a wiring target's mailboxes, or raise naming the missing id."""
        if target_node_id not in mailboxes:
            msg = f"Target node '{target_node_id}' not found in mailboxes."
            raise ValueError(msg)
        return mailboxes[target_node_id]
