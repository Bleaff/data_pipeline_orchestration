"""
Module contains a simple decorator for profiling of functions and methods.

Decorator is generating logging messages with information about function name,
arguments and result of the function call.

Example of usage:

.. code-block:: python

    @DetectorProfiler("my_detector")
    def my_detector(image):
        # do the detection
        pass

    my_detector(image)

"""

import logging
from functools import wraps

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class DetectorProfiler:
    """
    Decorator for profiling functions and methods.

    Decorator generates logging messages with information about:
        - function name
        - arguments
        - result of the function call

    Example of usage:

    .. code-block:: python

        @DetectorProfiler("my_detector")
        def my_detector(image):
            # do the detection
            pass

        my_detector(image)

    """

    def __init__(self, name=None):
        """
        Initialize the DetectorProfiler.

        Args:
            name (str, optional): Detector name which will be used in log messages.
                Defaults to None.
        """
        self.name = name

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wraps a function to log its arguments and result.

            Args:
                *args: Function arguments.
                **kwargs: Function keyword arguments.

            Returns:
                The result of the function call.
            """

            result = func(*args, **kwargs)
            LOGGER.debug(f"[{self.name}] Function: {func.__name__}, result: {result}, kwargs: {kwargs}")
            return result

        return wrapper
