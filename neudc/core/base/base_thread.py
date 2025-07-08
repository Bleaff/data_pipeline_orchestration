"""Abstract base class for threaded node implementations in a distributed system.

This module contains the base class for all threaded node implementations in a
distributed system. Threaded nodes are a special type of node that can be used to
run other nodes in separate threads, allowing for parallel execution of node
graphs.

The class provides an interface for threaded nodes, which must implement the
standard methods defined in the base class. The interface includes methods for
initializing the node, running the node's main functionality, and stopping the
node.
"""

from typing import Any

from neudc.core.base.base_node import BaseNode


class BaseThreadedNode(BaseNode):
    """Abstract base class for threaded node implementations in a distributed system.

    This class defines the standard methods a threaded node must implement to be used in
    the system. It provides a common interface for all nodes that run in a separate thread,
    regardless of the specific functionality they provide.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> "BaseThreadedNode":
        """Initialize the threaded node with a mailbox and a logger."""
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        """Start the node's thread."""
        self.init_runtime()
