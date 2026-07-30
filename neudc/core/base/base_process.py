"""Base node class that runs in a separate process.

This module contains the base class for all process nodes. Process nodes are a
special type of node that can be used to run other nodes in separate processes,
allowing for parallel execution of node graphs.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from neudc.core.base.base_node import BaseNode
from neudc.core.observability.metrics import HEALTH
from neudc.utils import LOGGER

mp.set_start_method("spawn", force=True)


class BaseProcessNode(BaseNode, mp.Process, ABC):
    """Base node class that runs in a separate process.

    Inherits from BaseNode and multiprocessing.Process.
    Includes health monitoring and lifecycle control.
    """

    HEALTH_INITIAL = False
    HEALTH_NORMAL = True
    HEALTH_CHECK_INTERVAL = 5  # seconds
    HEALTH_TIMEOUT = 15  # seconds

    def __init__(self, mailbox: Any, id: str = "BaseProcessNode") -> None:
        """Initialize the base process node.

        Args:
        ----
            mailbox: Mailbox for inter-process communication.
            id: Identifier for this node instance.

        """
        # Initialize BaseNode and multiprocessing.Process
        BaseNode.__init__(self, mailbox=None, id=id)
        # Capture a picklable snapshot of the mailbox, then explicitly release the
        # bound consume port in the parent so the child can rebind it after spawn.
        # (__getstate__ itself no longer stops the mailbox as a side effect.)
        self.mailbox_config = mailbox.__getstate__()
        mailbox.stop()
        mp.Process.__init__(self)
        # Health state lives in shared memory created here (like stop_event), so the
        # parent's status()/health monitor observe what the child writes. The child
        # MUST mutate these Values, never reassign them (see _mark_running).
        self._healthy = mp.Value("b", self.HEALTH_INITIAL)
        self._last_success_time = mp.Value("d", 0.0)
        self.stop_event = mp.Event()
        # BaseNode.__init__ created threading primitives. A process node must use the
        # multiprocessing ones instead: they are shared with the child, and a
        # threading.Event is not picklable for spawn in the first place.
        self._stop_event = self.stop_event
        self._failed_event = mp.Event()
        # Set by the child once its mailbox is rebound and runtime is ready; lets the
        # parent wait for real readiness instead of a fixed sleep before starting
        # upstream producers. Shared across spawn the same way as stop_event.
        self._ready_event = mp.Event()

    def start(self) -> None:
        """Start the process node."""
        LOGGER.info(f"Starting process node {self.id}...")
        mp.Process.start(self)

    def _start_afterwords(self) -> None:
        """Rebind the mailbox in the child process and start health monitoring.

        Does NOT start `BaseNode.init_runtime()`'s thread: `run()` below is already
        the process' single processing loop (collect -> process -> send). Starting
        both meant two consumers competing for the same mailbox (see #12).
        """
        self._health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self._health_thread.start()

        from neudc.core.communication.mailbox.zmq_mailbox import ZMQMailbox

        self.mailbox = ZMQMailbox.from_state(self.mailbox_config)

    def init_process_runtime(self) -> None:
        """Initialize runtime resources for the process node."""

    def run(self) -> None:
        """Process entrypoint: start health monitor and processing loop."""
        LOGGER.info(f"Starting process node {self.id}...(PID: {os.getpid()})")
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            # Metrics this child records live in its own per-pid files (see
            # neudc.entrypoints.main._maybe_start_prometheus); clean up on exit so a
            # dead node's last-known values don't linger forever in future scrapes.
            from prometheus_client import multiprocess

            atexit.register(multiprocess.mark_process_dead, os.getpid())
        self.init_process_runtime()
        self._mark_running()

        self._start_afterwords()  # Rebind mailbox, start health monitor
        self._ready_event.set()  # Mailbox is rebound and runtime is initialized.
        LOGGER.info(f"Process node started, entering processing loop...Mailbox status:{self.mailbox.consume_port}")

        while not self.stop_event.is_set():
            try:
                data = self._collect_data()
                if data is None:
                    continue

                # Same policy-aware path as the threaded loop (see BaseNode._handle):
                # retries, dead-lettering and the fail action live in one place.
                # `on_item=self._send_item` streams a generator process() (#37) out as
                # each value is yielded, instead of waiting for the whole thing.
                result = self._handle(data, on_item=self._send_item)
                # Shared with BaseNode._run so the two loops cannot drift apart (#40, #12).
                if self._should_send(result):
                    self._send_item(result)
                # update last success time
                with self._last_success_time.get_lock():
                    self._last_success_time.value = time.time()
            except Exception as e:  # noqa: BLE001 -- top-level process run-loop must never crash the process.
                LOGGER.exception("[BaseProcessNode] Exception", exc_info=e)

    def _send_item(self, item: Any) -> None:
        """Send one item to the mailbox and update the health-monitor's liveness timestamp.

        Unlike `BaseNode._send_item`, a send failure here is **not** caught locally —
        it propagates to this loop's own `except Exception` above, exactly as a
        classic single-result send failure already did (and, matching that existing
        behaviour, `_last_success_time` is then *not* updated for this iteration).
        """
        self.mailbox.send(item)
        with self._last_success_time.get_lock():
            self._last_success_time.value = time.time()

    def _mark_running(self) -> None:
        """Mark the node healthy and reset its last-success timestamp (child side).

        Mutates the shared mp.Value objects created in __init__ instead of replacing
        them, so the parent's status() and the health monitor see the child's state.
        """
        with self._healthy.get_lock():
            self._healthy.value = self.HEALTH_NORMAL
        with self._last_success_time.get_lock():
            self._last_success_time.value = time.time()
        HEALTH.labels(node=self.id).set(1)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the child signals readiness (mailbox rebound, runtime up).

        Returns True if the node became ready within the timeout, else False.
        """
        return self._ready_event.wait(timeout)

    def stop(self) -> None:
        """Signal the process to stop and wait for the health thread."""
        LOGGER.info("Stopping process node...")
        self.stop_event.set()

    def status(self) -> bool:
        """Return current health status (True = healthy)."""
        with self._healthy.get_lock():
            return bool(self._healthy.value)

    def _health_monitor(self) -> None:
        """Monitor if node continues to process over time."""
        while not self.stop_event.is_set():
            time.sleep(self.HEALTH_CHECK_INTERVAL)
            with self._last_success_time.get_lock():
                delta = time.time() - self._last_success_time.value
            if delta > self.HEALTH_TIMEOUT:
                LOGGER.warning(f"Health {self.__class__.__name__} timeout exceeded!")
                with self._healthy.get_lock():
                    self._healthy.value = False
                HEALTH.labels(node=self.id).set(0)

    @abstractmethod
    def process(self, item: Any) -> Any:
        """Core logic of the node. Must be implemented."""

    @classmethod
    def from_config(cls: type[BaseProcessNode], config: dict[str, Any]) -> BaseProcessNode:
        """From config-based constructor for building node with specified config."""
        raise NotImplementedError

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the process node."""
        # Update the instance's __dict__ with the state dictionary
        # This is necessary for unpickling the process node
        self.__dict__.update(state)
