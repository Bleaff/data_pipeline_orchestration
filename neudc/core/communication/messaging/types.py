"""Message types carried by the graph.

Everything that travels between nodes is a :class:`BaseMessage` — the common
envelope — or a :class:`Batch` of them. :class:`Frame` is the CV payload
(pixels plus annotations); other modalities subclass ``BaseMessage`` the same
way, so mailboxes and node contracts are typed by the envelope rather than by
images.

Also defines the attributes that nodes annotate onto a frame: boxes, classes,
segmentation masks, keypoints.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np  # noqa: TC002 -- used as a real pydantic field type, must stay importable at runtime
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterator

# === Base Attribute ===


class BaseAttribute(BaseModel):
    """Base class for box attributes."""

    source_node_id: int = -1


# === Specific Attributes ===


class Class(BaseAttribute):
    """Attribute representing class membership."""

    class_id: str
    score: float | None = None


class Segmentation(BaseAttribute):
    """Segmentation mask (image) attached to a box."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray


class Text(BaseAttribute):
    """Text attribute with confidence."""

    text: str
    score: float


# === Detection Box ===


class Box(BaseAttribute):
    """Box with absolute coordinates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x1: float | int
    y1: float | int
    x2: float | int
    y2: float | int
    class_id_name: str
    class_id_num: int
    score: float
    labels: list[Class] | None = None
    reid: str = "-1"


# === Keypoints ===


class Keypoint(BaseModel):
    """Relative keypoint (0.0 - 1.0)."""

    x: float
    y: float
    class_id: str
    score: float


class Keypoints(BaseModel):
    """Dict[str, Keypoint]. Used by models to describe body parts, facial landmarks, etc."""

    keypoints: dict[str, Keypoint] = {}
    description: str = ""


# === Messages ===


class BaseMessage(BaseModel):
    """Common envelope for anything that travels between nodes.

    Concrete payloads subclass this: :class:`Frame` for the CV path, and the
    text / audio / token schemas for other modalities. Mailboxes and node
    contracts are typed by this class, so the graph is not tied to images.

    Large buffers on subclasses (pixels, audio samples, tensors) need no special
    handling here — the codec pulls them out of the pickle stream generically,
    see :mod:`neudc.core.communication.messaging.codec`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float
    source: str = ""
    """Who produced the message: node id, device or stream name.

    Distinct from :attr:`Frame.source_frame`, which is the path of an image on
    disk and stays CV-specific.
    """

    drop: bool = False
    """Marks a message the downstream nodes should skip rather than consume."""

    # Streaming identity. Unused by the batch labelling path, where every message
    # stands alone; a streaming producer sets these so a consumer can tell which
    # dialogue/turn a message belongs to and when that turn ended.
    session_id: str | None = None
    turn_id: int | None = None
    is_final: bool = True


class ControlAction(StrEnum):
    """Action carried by a :class:`ControlMessage`."""

    CANCEL = "cancel"


class ControlMessage(BaseMessage):
    """A priority control-plane message (e.g. turn cancellation / barge-in).

    Delivered out-of-band from the data FIFO — over the mailbox's dedicated control
    channel (see :class:`~neudc.core.communication.mailbox.zmq_mailbox.ZMQMailbox`) —
    so it is not stuck behind a backlog of data messages it is meant to interrupt.
    """

    action: ControlAction


class Frame(BaseMessage):
    """Frame with image, timestamp, and boxes."""

    image: np.ndarray
    source_frame: str
    frame_id: int
    boxes: list[Box]
    embedding: np.ndarray | None = None
    frame_id_last: int


# === Non-CV payloads (#34) ===


class TextChunk(BaseMessage):
    """A chunk of text: an ASR partial/final hypothesis, an LLM token stream chunk, etc."""

    text: str
    language: str | None = None


class TokenTensor(BaseMessage):
    """Token ids or logits from an LLM/tokenizer, as a tensor."""

    data: np.ndarray


class AudioChunk(BaseMessage):
    """A chunk of PCM audio samples."""

    samples: np.ndarray
    sample_rate: int
    channels: int = 1


class VideoSegment(BaseMessage):
    """A short run of raw video frames traveling as one message (pre-CV-processing).

    Distinct from ``Batch[Frame]``: a batch is unrolled by the mailbox into individual
    edge writes, while a ``VideoSegment`` is one atomic streaming unit (e.g. a fixed-size
    window from a live source), matching how ``AudioChunk`` groups several audio samples
    into one message.
    """

    images: list[np.ndarray]
    start_frame_id: int
    fps: float | None = None


MessageT = TypeVar("MessageT", bound=BaseMessage)


class Batch(BaseModel, Generic[MessageT]):
    """Group of messages handled as one unit (batched inference, selection).

    Generic over the payload: ``Batch[Frame]`` for the CV path. The field keeps
    the name ``frames`` for compatibility with existing nodes.

    A batch is not itself a :class:`BaseMessage`: the mailbox unrolls it and sends
    its items one by one, so a batch never crosses an edge as a unit.

    That is what makes one pydantic quirk harmless — a parametrized alias built
    inside a function (``Batch[Frame](...)`` in a node method) is **not picklable**,
    because pydantic only registers the generated class in its module when the
    parametrization happens at module level. Batches stay process-local, so this
    never reaches the transport; serialize the unparametrized ``Batch`` if you ever
    need one on the wire.
    """

    frames: list[MessageT]

    def __iter__(self) -> Iterator[MessageT]:  # type: ignore[override]
        """Iterate over the messages in this batch (intentionally shadows BaseModel.__iter__)."""
        return iter(self.frames)

    def __len__(self) -> int:
        """Return the number of messages in this batch."""
        return len(self.frames)

    def __getitem__(self, index: int) -> MessageT:
        """Return the message at ``index``."""
        return self.frames[index]
