"""Message (de)serialization for the ZeroQueue transport.

Uses pickle protocol 5, which supports out-of-band buffers and serializes NumPy
arrays (the bulk of a Frame) noticeably faster than the default protocol. Keeping
the codec in one place lets the mailbox serialize a message **once** and hand the
same bytes to every fan-out edge, instead of re-pickling per downstream node.

Note on zero-copy: a true zero-copy / shared-memory image transport is intentionally
not done here. Several nodes (e.g. DrawNode) mutate ``frame.image`` in place, so a
read-only buffer view shared across processes would be unsafe. That optimization
needs explicit shared-memory ownership and is left as a separate, larger change.
"""

from __future__ import annotations

import pickle
from typing import Any

# Protocol 5 (PEP 574): out-of-band buffers, efficient for large NumPy arrays.
_PROTOCOL = 5


def dumps(obj: Any) -> bytes:
    """Serialize a message object to bytes."""
    return pickle.dumps(obj, protocol=_PROTOCOL)


def loads(raw: bytes) -> Any:
    """Deserialize bytes produced by :func:`dumps` back into an object."""
    return pickle.loads(raw)
