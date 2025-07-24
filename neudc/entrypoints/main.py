"""Script serves as the entry point for the application.

running the system from config file:
nodes:
  - id: reader
    type: FolderImageNode
    folder_path: "assets/images"
    mode: "loop"
    frame_delay: 0.1
    outputs: [hash]

  - id: hash
    type: HashNode
    delta: 5
    hash_size: 8
    hash_type: 'ahash'
    outputs: [model]

  - id: model
    type: ProcessDetInference
    model_config:
      type: YOLOv8
      path: "/home/user/mkotovanu/neudc/models/yolov8n.torchscript"
      backend: TorchBackend
      device_id: 0
    outputs: [visualizer]

  - id: visualizer
    type: DrawNode
    outputs: [saver]

  - id: saver
    type: SaveImageNode
    save_dir: "./output_images"
    outputs: []


It initializes and configures the necessary components for the system to run,
including the routing and node factories. The configuration is loaded from a specified
YAML file, which contains the details for setting up the nodes and their connections.

Functions:
- main(config_path: str) -> None: Initiates the application with the given configuration file path.
"""

import time

from prometheus_client import start_http_server

from neudc.core.base import BaseProcessNode, BaseThreadedNode
from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.node_factory import NodeFactory
from neudc.core.utils.config_loader import load_config
from neudc.utils.logger import LOGGER


def main(config_path: str) -> None:
    """Entrypoint for the application. Takes the path to the YAML config as input. Factories nodes and routes messages between them.

    Args:
    ----
        config_path (str): Path to the YAML configuration file.

    """
    is_running = True

    # Load pipeline configuration from YAML
    config = load_config(config_path)

    # Load metrics
    prometheus_config = config.get("prometheus")
    if prometheus_config and prometheus_config.get("port") and prometheus_config["enable"]:
        start_http_server(prometheus_config["port"])

    # 1. Create routing factory and initialize mailboxes
    router = RoutingFactory(config)
    mailbox_map = router.create_mailboxes()

    # 2. Create node instances using NodeFactory
    nodes = []

    # 1. Сначала запусти процессные
    for node_config in config["nodes"]:
        node_id = node_config["id"]
        mailbox = mailbox_map[node_id]
        node = NodeFactory.create(node_config, mailbox=mailbox)
        if isinstance(node, BaseProcessNode):
            node.start()
            nodes.append(node)
    time.sleep(10)
    LOGGER.info("All processes started. Now time to fucking wait for x seconds.")
    # 2. Потом запусти тредовые
    for node_config in config["nodes"]:
        node_id = node_config["id"]
        mailbox = mailbox_map[node_id]
        node = NodeFactory.create(node_config, mailbox=mailbox)
        if isinstance(node, BaseThreadedNode):
            node.start()
            nodes.append(node)

    # 3. Keep the main thread alive while nodes are working
    try:
        while is_running:
            pass
    except KeyboardInterrupt:
        is_running = False
        for node in nodes:
            node.stop()


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    main(config_file)
