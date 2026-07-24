"""Single-pipeline entry point for the application.

Running the system from a config file:

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
      path: "/path/to/yolov8n.torchscript"
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

It loads the YAML config, builds the routing/mailboxes and node instances, starts
process nodes first (waiting for each to signal readiness so their mailboxes are
bound before upstream producers start), then starts threaded nodes, and finally
blocks until a shutdown signal arrives.
"""

from __future__ import annotations

import signal
import threading

from prometheus_client import start_http_server

from neudc.core.base import BaseProcessNode
from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.node_factory import NodeFactory
from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import validate_pipeline_config
from neudc.utils.logger import LOGGER

# How long to wait for a process node to report readiness before starting producers.
STARTUP_READY_TIMEOUT_SEC = 30.0

# How often to check whether a node gave up under an `on_error: fail` policy.
FAILURE_POLL_INTERVAL_SEC = 0.5


def _maybe_start_prometheus(config: dict) -> None:
    """Start the Prometheus HTTP server if enabled in the config."""
    prometheus_config = config.get("prometheus")
    if prometheus_config and prometheus_config.get("port") and prometheus_config.get("enable"):
        start_http_server(prometheus_config["port"])
        LOGGER.info(f"Prometheus metrics server started on port {prometheus_config['port']}")


def main(config_path: str) -> None:
    """Build and run a single pipeline from a YAML configuration file.

    Args:
    ----
        config_path (str): Path to the YAML configuration file.

    """
    config = load_config(config_path)
    validate_pipeline_config(config)  # fail fast with a clear message on a bad config
    _maybe_start_prometheus(config)

    # 1. Routing: one mailbox per node, wired to their outputs.
    router = RoutingFactory(config)
    mailbox_map = router.create_mailboxes()

    # 2. Build every node exactly once and split by execution model.
    process_nodes: list = []
    threaded_nodes: list = []
    for node_config in config["nodes"]:
        node = NodeFactory.create(node_config, mailbox=mailbox_map[node_config["id"]])
        (process_nodes if isinstance(node, BaseProcessNode) else threaded_nodes).append(node)

    all_nodes = process_nodes + threaded_nodes

    # 3. Start process nodes first and wait until each has bound its mailbox,
    #    so no producer sends into a not-yet-listening consumer.
    for node in process_nodes:
        node.start()
    for node in process_nodes:
        if not node.wait_ready(timeout=STARTUP_READY_TIMEOUT_SEC):
            LOGGER.warning(f"Process node '{node.id}' did not report readiness in time; starting anyway.")

    # 4. Start threaded nodes (readers/producers/light processors).
    for node in threaded_nodes:
        node.start()

    LOGGER.info(f"Pipeline running with {len(all_nodes)} nodes. Waiting for shutdown signal...")

    # 5. Block until interrupted, without a busy loop.
    stop_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        LOGGER.info(f"Received signal {signum}; shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # 6. A node whose error policy is `fail` only flags itself; bringing the pipeline
    #    down is the owner's job, which here is this entry point.
    def _watch_for_failed_nodes() -> None:
        while not stop_event.wait(FAILURE_POLL_INTERVAL_SEC):
            broken = [node.id for node in all_nodes if node.failed()]
            if broken:
                LOGGER.critical(f"Node(s) {broken} failed under their error policy; shutting down.")
                stop_event.set()
                return

    threading.Thread(target=_watch_for_failed_nodes, daemon=True).start()

    try:
        stop_event.wait()
    finally:
        for node in all_nodes:
            node.stop()
        LOGGER.info("All nodes stopped.")


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    main(config_file)
