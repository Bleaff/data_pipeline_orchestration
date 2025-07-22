from abc import ABC, abstractmethod

from prometheus_client import Counter, Gauge


class BaseProfiler(ABC):

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        metric_prefix = f"{self.node_name}".replace(" ", "_")
        self.exec_time_gauge = Gauge(f"{metric_prefix}_execution_time_seconds", "Execution time of the function")
        self.total_frames_counter = Counter(f"{metric_prefix}_frames_total", "Total number of frames")
        self.filtered_frames_counter = Counter(f"{metric_prefix}_frames_filtered", "Number of filtered frames")
        self.go_through_counter = Counter(f"{metric_prefix}_frames_passed", "Number of passed frames")

    @abstractmethod
    def __call__(self, func):
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
