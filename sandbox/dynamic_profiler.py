import logging
from functools import wraps

from decorate_me import DetectorProfiler

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def dynamic_profiler(method):
    """Function decorator that wraps a given method in a DetectorProfiler instance.

    DetectorProfiler is a class that provides a simple way to profile the execution time of a given function.
    It uses the time.perf_counter() function to measure the time difference between the start and end of the function call.
    The profile is then logged with the name of the function.

    Args:
        method: The method to be profiled.

    Returns:
        A decorated method that wraps the given method in a DetectorProfiler instance.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        profiler = DetectorProfiler(name=self.name)
        wrapped_method = profiler(method)
        return wrapped_method(self, *args, **kwargs)

    return wrapper
