import logging
from functools import wraps

from neudc.utils.profile import Profile

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class DetectorProfiler(Profile):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            LOGGER.debug(f"For function: {func.__name__}, result: {result}, arguments: {args, kwargs}")
            return result

        return wrapper
