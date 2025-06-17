"""Module provides the foundational classes and functions for the application.

It defines core data structures, utility methods, and base logic that are shared
across other modules. Import this module to access essential building blocks
required for extending or customizing the application's functionality.
"""

from .base_mailbox import BaseMailbox
from .base_process import BaseProcessNode
from .base_queue import QueueLike
from .base_thread import BaseThreadedNode

__all__ = [
    "BaseThreadedNode",
    "BaseProcessNode",
    "BaseMailbox",
    "QueueLike",
]
