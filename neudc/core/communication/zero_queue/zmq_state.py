"""Module contains the state of the ZeroQueue, such as the mode of the current queue."""

from enum import StrEnum


class ZeroQueueMode(StrEnum):
    """Enum for ZeroQueue modes."""

    SUB = "SUB"
    PUB = "PUB"


class ZeroQueueConnectionType(StrEnum):
    """Enum for ZeroQueue connection states.

    Connecting to an existing port or binding(hosting) to a new one.

    fields:
        CONNECT
        BIND

    values:
        "connect"
        "bind"
    """

    CONNECT = "connect"
    BIND = "bind"
