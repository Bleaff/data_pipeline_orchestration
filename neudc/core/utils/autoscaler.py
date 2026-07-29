"""Queue-depth autoscaler for a node's worker replicas (#15).

Orchestration logic that coordinates several running node instances of one logical
node — not a transport concern (``core/communication``) and not an interface
(``core/base``), so it lives here alongside :class:`~neudc.core.communication.messaging.routing_factory.RoutingFactory`
and the rest of the config/wiring utilities.

``ReplicaAutoscaler`` is deliberately generic over the replica type: it takes a
starting list of replicas, an :class:`~neudc.core.utils.autoscale.AutoscaleConfig`,
and three callables — how to read one replica's current queue depth, how to create a
new replica, and how to tear one down. This keeps the decision engine (``check_and_scale``)
fully testable with fakes/mocks, independent of whether a "replica" is a real
``BaseNode`` with a live ZMQ mailbox or a lightweight stand-in in a unit test.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

from neudc.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable

    from neudc.core.utils.autoscale import AutoscaleConfig

T = TypeVar("T")


class ReplicaAutoscaler(Generic[T]):
    """Scales a node's replica pool up or down by one, based on queue depth.

    Scale-up trigger: **every** current replica's queue depth is at or above
    ``queue_depth_high`` and the pool is below ``max_replicas``.
    Scale-down trigger: **every** current replica's queue depth is at or below
    ``queue_depth_low`` and the pool is above ``min_replicas``.
    Outside those bounds — or exactly at the current pool size limits — no action is
    taken. Only one replica is added/removed per call, keeping each step small and
    observable rather than jumping straight to some target size.
    """

    def __init__(
        self,
        replicas: list[T],
        config: AutoscaleConfig,
        queue_depth_reader: Callable[[T], int],
        scale_up: Callable[[], T],
        scale_down: Callable[[T], None],
    ) -> None:
        """Initialize the autoscaler.

        Args:
        ----
            replicas (list[T]): The currently running replicas for one logical node.
                Mutated in place as ``check_and_scale`` adds/removes entries.
            config (AutoscaleConfig): Thresholds and bounds (validated at config load time).
            queue_depth_reader (Callable[[T], int]): Returns one replica's current queue depth.
            scale_up (Callable[[], T]): Creates, starts and returns one new replica.
            scale_down (Callable[[T], None]): Stops and tears down one replica.

        """
        self._replicas: list[T] = list(replicas)
        self.config = config
        self._queue_depth_reader = queue_depth_reader
        self._scale_up = scale_up
        self._scale_down = scale_down
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def replicas(self) -> list[T]:
        """Current replica pool (a copy; mutate only through scaling)."""
        return list(self._replicas)

    def check_and_scale(self) -> str | None:
        """Inspect queue depth once and scale by at most one replica if warranted.

        Returns
        -------
            str | None: ``"up"``, ``"down"``, or ``None`` if no scaling happened.
                Mainly useful for tests and logging to assert on the outcome.

        """
        with self._lock:
            if not self._replicas:
                return None
            depths = [self._queue_depth_reader(r) for r in self._replicas]

            if (
                all(d >= self.config.queue_depth_high for d in depths)
                and len(self._replicas) < self.config.max_replicas
            ):
                new_replica = self._scale_up()
                self._replicas.append(new_replica)
                LOGGER.info(
                    f"Autoscale: scaled up to {len(self._replicas)} replicas "
                    f"(all queue depths >= {self.config.queue_depth_high})"
                )
                return "up"

            if all(d <= self.config.queue_depth_low for d in depths) and len(self._replicas) > self.config.min_replicas:
                victim_idx = min(range(len(self._replicas)), key=lambda i: depths[i])
                victim = self._replicas.pop(victim_idx)
                self._scale_down(victim)
                LOGGER.info(
                    f"Autoscale: scaled down to {len(self._replicas)} replicas "
                    f"(all queue depths <= {self.config.queue_depth_low})"
                )
                return "down"

            return None

    def start(self) -> None:
        """Start a background thread calling ``check_and_scale`` every ``check_interval_s``."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.config.check_interval_s):
            try:
                self.check_and_scale()
            except Exception as e:  # noqa: BLE001 -- background loop must never crash the thread.
                LOGGER.exception("Error during autoscale check", exc_info=e)

    def stop(self) -> None:
        """Stop the background thread, if running."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
