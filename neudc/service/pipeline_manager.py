"""
Manager for handling pipelines in a multiprocessing environment.

This class provides methods to start, stop, and check the status of pipelines. Each pipeline
is run in a separate process, allowing for parallel execution. The manager maintains a dictionary
of currently running pipelines, each associated with a stop event for graceful termination.

Fields:
    pipelines (dict): A dictionary mapping pipeline names to tuples of (BasePipeline, mp.Event).

Methods:
    start_pipeline(name, nodes_config, task_config): Starts a new pipeline with the given name and configuration.
    stop_pipeline(name): Stops the specified pipeline if it is currently running.
    status(): Returns the current status (running or stopped) of all managed pipelines.
"""

import multiprocessing as mp
from neudc.core.base.base_pipeline import BasePipeline

class PipelineServiceManager:
    """Manager for handling pipelines in a multiprocessing environment.
    This class provides methods to start, stop, and check the status of pipelines. Each pipeline
    is run in a separate process, allowing for parallel execution. The manager maintains a dictionary
    of currently running pipelines, each associated with a stop event for graceful termination.
    """
    def __init__(self):
        """Initialize the PipelineServiceManager."""
        self.pipelines: dict[str, tuple[BasePipeline, mp.Event]] = {}

    def start_pipeline(self, name: str, nodes_config: list[dict], task_config: dict = {}):
        """Starts a new pipeline with the given name and configuration."""
        if name in self.pipelines:
            raise ValueError(f"Pipeline '{name}' already running")

        stop_event = mp.Event()
        pipeline = BasePipeline(name=name, stop_event=stop_event, nodes_config=nodes_config, task_config=task_config)
        pipeline.start()
        self.pipelines[name] = (pipeline, stop_event)

    def stop_pipeline(self, name: str):
        """Stops the specified pipeline if it is currently running."""
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not found")
        
        pipeline, stop_event = self.pipelines.pop(name)
        stop_event.set()
        pipeline.join(timeout=5)
        if pipeline.is_alive():
            pipeline.terminate()

    def status(self):
        """Returns the current status (running or stopped) of all managed pipelines."""
        return {
            name: ("running" if pipe.is_alive() else "stopped")
            for name, (pipe, _) in self.pipelines.items()
        }
