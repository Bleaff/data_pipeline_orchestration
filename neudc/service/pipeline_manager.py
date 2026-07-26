"""Manager for handling pipelines in a multiprocessing environment.

This class provides methods to start, stop, and check the status of pipelines. Each pipeline
is run in a separate process, allowing for parallel execution. The manager maintains a dictionary
of currently running pipelines, each associated with a stop event for graceful termination.

Fields:
    pipelines (dict): A dictionary mapping pipeline names to tuples of (BasePipeline, stop event).

Methods:
    start_pipeline(name, nodes_config, task_config): Starts a new pipeline with the given name and configuration.
    stop_pipeline(name): Stops the specified pipeline if it is currently running.
    status(): Returns the current status (running or stopped) of all managed pipelines.

"""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

from neudc.core.base.base_pipeline import BasePipeline

if TYPE_CHECKING:
    from neudc.core.base.base_node import EventLike


class PipelineServiceManager:
    """Manager for handling pipelines in a multiprocessing environment.

    This class provides methods to start, stop, and check the status of pipelines. Each pipeline
    is run in a separate process, allowing for parallel execution. The manager maintains a dictionary
    of currently running pipelines, each associated with a stop event for graceful termination.
    """

    def __init__(self) -> None:
        """Initialize the PipelineServiceManager."""
        self.pipelines: dict[str, tuple[BasePipeline, EventLike]] = {}

    def start_pipeline(
        self, name: str, nodes_config: list[dict[str, Any]], task_config: dict[str, Any] | None = None
    ) -> None:
        """Start a new pipeline with the given name and configuration.

        Args:
        ----
            name (str): Unique name identifying the pipeline.
            nodes_config (list[dict[str, Any]]): Per-node configuration entries for the pipeline.
            task_config (dict[str, Any] | None): Task-level configuration shared across nodes.
                Defaults to an empty dict.

        Raises:
        ------
            ValueError: If a pipeline with ``name`` is already running.

        """
        if name in self.pipelines:
            msg = f"Pipeline '{name}' already running"
            raise ValueError(msg)

        stop_event = mp.Event()
        pipeline = BasePipeline(
            name=name, stop_event=stop_event, nodes_config=nodes_config, task_config=task_config or {}
        )
        pipeline.start()
        self.pipelines[name] = (pipeline, stop_event)

    def stop_pipeline(self, name: str) -> None:
        """Stop the specified pipeline if it is currently running.

        Args:
        ----
            name (str): Name of the pipeline to stop.

        Raises:
        ------
            ValueError: If no pipeline with ``name`` is currently managed.

        """
        if name not in self.pipelines:
            msg = f"Pipeline '{name}' not found"
            raise ValueError(msg)

        pipeline, stop_event = self.pipelines.pop(name)
        stop_event.set()
        pipeline.join(timeout=5)
        if pipeline.is_alive():
            pipeline.terminate()

    def status(self) -> dict[str, str]:
        """Return the current status (running or stopped) of all managed pipelines."""
        return {name: ("running" if pipe.is_alive() else "stopped") for name, (pipe, _) in self.pipelines.items()}
