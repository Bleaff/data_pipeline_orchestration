"""Multi-pipeline entry point: runs every pipe declared under a config's `pipes` key as a service."""

from __future__ import annotations

import logging
import signal
import time
from typing import TYPE_CHECKING

from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import ConfigError, validate_pipeline_config
from neudc.service.pipeline_manager import PipelineServiceManager

if TYPE_CHECKING:
    from types import FrameType


def main(config_path: str) -> None:
    """Run every pipeline declared under the config's `pipes` key as a managed service.

    Args:
    ----
        config_path (str): Path to the YAML configuration file.

    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    config = load_config(config_path)
    pipes_config = config.get("pipes", [])

    if not pipes_config:
        logger.error("No 'pipes' found in configuration.")
        return

    manager = PipelineServiceManager()

    # Graceful shutdown hook
    is_stopping = False

    def shutdown_handler(_signum: int | None, _frame: FrameType | None) -> None:
        nonlocal is_stopping
        if is_stopping:
            return
        is_stopping = True
        logger.info("Shutdown signal received. Stopping pipelines...")
        for name in list(manager.pipelines.keys()):
            manager.stop_pipeline(name)
        logger.info("All pipelines stopped.")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start all pipelines
    for pipe in pipes_config:
        name = pipe["name"]
        nodes = pipe["nodes"]
        task = pipe.get("task", {})
        try:
            validate_pipeline_config({"nodes": nodes})  # fail fast on a bad pipe
            manager.start_pipeline(name=name, nodes_config=nodes, task_config=task)
            logger.info(f"Started pipeline: {name}")
        except ConfigError:
            logger.exception(f"Invalid config for pipeline '{name}'")
        except Exception:  # best-effort startup loop: one bad pipe must not abort the others.
            logger.exception(f"Failed to start pipeline {name}")

    # Keep alive
    try:
        while not is_stopping:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/pipes.yaml"
    main(config_path)
