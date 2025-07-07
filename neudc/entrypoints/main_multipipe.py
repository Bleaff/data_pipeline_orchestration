import logging
import time
import signal
from neudc.core.utils.config_loader import load_config
from neudc.service.pipeline_manager import PipelineServiceManager


def main(config_path: str) -> None:
    """Main entrypoint to run multiple pipelines as services."""
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
    def shutdown_handler(signum, frame):
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
            manager.start_pipeline(name=name, nodes_config=nodes, task_config=task)
            logger.info(f"Started pipeline: {name}")
        except Exception as e:
            logger.exception(f"Failed to start pipeline {name}: {e}")

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
