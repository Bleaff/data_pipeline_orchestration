import logging
from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.node_factory import NodeFactory
from neudc.core.utils.config_loader import load_config


def main(config_path: str):
    is_running = True
    logging.basicConfig(level=logging.DEBUG)

    # Load pipeline configuration from YAML
    config = load_config(config_path)

    # 1. Create routing factory and initialize mailboxes
    router = RoutingFactory(config)
    mailbox_map = router.create_mailboxes()

    # 2. Create node instances using NodeFactory
    nodes = []
    for node_config in config["nodes"]:
        node_id = node_config["id"]
        mailbox = mailbox_map[node_id]
        logger = logging.getLogger(node_id)
        node = NodeFactory.create(node_config, mailbox=mailbox, logger=logger)
        nodes.append(node)

    # 3. Keep the main thread alive while nodes are working
    try:
        while is_running:
            pass
    except KeyboardInterrupt:
        print("\n[Main] Stopping all nodes...")
        is_running = False
        for node in nodes:
            node.stop()


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    main(config_file)
