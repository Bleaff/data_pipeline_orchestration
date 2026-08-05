"""Manager for handling pipelines in a multiprocessing environment.

This class provides methods to start, stop, and check the status of pipelines. Each pipeline
is run in a separate process, allowing for parallel execution. The manager maintains a dictionary
of currently running pipelines, each associated with a stop event for graceful termination.

Fields:
    pipelines (dict): A dictionary mapping pipeline names to tuples of (BasePipeline, stop event).
    external_pipelines (dict): Pipelines this manager doesn't own the process lifecycle of — see
        `register_external_pipeline`.

Methods:
    start_pipeline(name, nodes_config, task_config): Starts a new pipeline with the given name and configuration.
    stop_pipeline(name): Stops the specified pipeline if it is currently running.
    register_external_pipeline(name, nodes_config): Makes a caller-owned pipeline visible to status()/the dashboard.
    unregister_external_pipeline(name): Undoes register_external_pipeline.
    status(): Returns the current status (running or stopped) of all managed pipelines.

"""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Literal

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
        # Pipelines this manager does NOT own the process lifecycle of: e.g. one with
        # live device objects (AudioReaderNode/AudioPlayerNode's mic/speaker) that
        # can't be expressed in start_pipeline's JSON contract, so the caller builds
        # and runs it itself (see docs/voice_assistant/README.md). Registered here
        # purely so GET /pipelines and /pipelines/{name} can see it — this manager
        # never starts, stops, or health-checks it.
        self.external_pipelines: dict[str, list[dict[str, Any]]] = {}

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
            ValueError: If a pipeline with ``name`` is already running (managed or external).

        """
        if name in self.pipelines or name in self.external_pipelines:
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
            ValueError: If no *managed* pipeline with ``name`` exists — an external one
                (see `register_external_pipeline`) has no process here to stop; use
                `unregister_external_pipeline` instead.

        """
        if name not in self.pipelines:
            msg = f"Pipeline '{name}' not found"
            raise ValueError(msg)

        pipeline, stop_event = self.pipelines.pop(name)
        stop_event.set()
        pipeline.join(timeout=5)
        if pipeline.is_alive():
            pipeline.terminate()

    def register_external_pipeline(self, name: str, nodes_config: list[dict[str, Any]]) -> None:
        """Make a caller-owned pipeline visible to `status()`/the dashboard, without managing it.

        For a pipeline this manager structurally can't start via `start_pipeline` — its
        config isn't JSON-only, e.g. `AudioReaderNode`/`AudioPlayerNode` need a live
        `device` object injected before `NodeFactory.create` (see
        `docs/voice_assistant/README.md`). The caller starts/stops the actual node
        threads/process itself; this only records the node list so `GET /pipelines`
        and `/pipelines/{name}` can show it, and reports it as always "running" for as
        long as it stays registered (there is no process here to poll `is_alive()` on).

        Args:
        ----
            name (str): Unique name identifying the pipeline.
            nodes_config (list[dict[str, Any]]): Per-node configuration entries, for
                display only — must be JSON-safe (no injected live device objects).

        Raises:
        ------
            ValueError: If a pipeline with ``name`` is already running (managed or external).

        """
        if name in self.pipelines or name in self.external_pipelines:
            msg = f"Pipeline '{name}' already running"
            raise ValueError(msg)
        self.external_pipelines[name] = nodes_config

    def unregister_external_pipeline(self, name: str) -> None:
        """Undo `register_external_pipeline`. Does not stop or otherwise touch the caller's nodes.

        Args:
        ----
            name (str): Name previously passed to `register_external_pipeline`.

        Raises:
        ------
            ValueError: If no external pipeline with ``name`` is registered.

        """
        if name not in self.external_pipelines:
            msg = f"External pipeline '{name}' not found"
            raise ValueError(msg)
        del self.external_pipelines[name]

    def status(self) -> dict[str, Literal["running", "stopped"]]:
        """Return the current status (running or stopped) of all managed and external pipelines."""
        statuses: dict[str, Literal["running", "stopped"]] = {
            name: ("running" if pipe.is_alive() else "stopped") for name, (pipe, _) in self.pipelines.items()
        }
        statuses.update(dict.fromkeys(self.external_pipelines, "running"))
        return statuses
