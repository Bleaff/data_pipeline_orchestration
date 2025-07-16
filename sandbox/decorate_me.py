from functools import wraps

from neudc.utils.profile import Profile


class DetectorProfiler(Profile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            return func(*args, **kwargs)

        return wrapper
