from functools import wraps
from contextlib import ContextDecorator
from typing import Callable, Any, TypeVar
from neudc.utils.profile import Profile, NoProfile, get_profile, conditional_jit, toggle_jit

class DetectorProfiler(Profile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_name = None
    