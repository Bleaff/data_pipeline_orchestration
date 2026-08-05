"""Node-type catalog for the visual pipeline config builder (#24).

`NODE_TYPE_CATALOG` is hand-curated from each node's `from_config` (rather than
introspected via `inspect.signature`) for two reasons: `from_config` reads config
dict keys directly (`config["x"]` / `config.get("x", default)`), which isn't
recoverable from a method signature; and introspecting every registered class would
mean eagerly importing all of them (torch, ultralytics, ...) just to list a catalog,
defeating `NodeFactory`'s lazy-import design. `get_node_catalog()` fills in
`accepts`/`emits` dynamically via `NodeFactory` (already lazy, and already paid for by
`config_schema`'s own validation) so those two stay a single source of truth.

Several things are deliberately left out of the builder: `AudioReaderNode` and
`AudioPlayerNode` (their `device` field is a live Python object, not YAML/JSON-expressible)
and the `SAHIDetector` model type (its `detector` field is likewise a live `YOLOv8`
instance). `AsrNode`/`LlmNode`/`TtsNode` are also not yet listed: their config
(`asr_config`/`llm_config`/`tts_config`) is a nested mapping, and `FieldKind` has no
"nested object" variant yet to render one — adding that is separate work. All of these
remain usable by hand-writing YAML directly; the catalog is additive tooling, not the
only way to build a config.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from neudc.core.node.node_factory import NodeFactory

__all__ = ("NodeTypeSpec", "get_node_catalog")

FieldKind = Literal["string", "integer", "number", "boolean", "enum", "model_config"]


class NodeFieldSpec(BaseModel):
    """One editable config key for a node type (or a model type's sub-config)."""

    name: str
    kind: FieldKind
    required: bool = False
    default: Any = None
    description: str = ""
    options: list[str] | None = None  # only meaningful when kind == "enum"


class ModelTypeSpec(BaseModel):
    """Sub-config fields for one `model_config.type` choice (YOLOv8, BlurClassification, ...)."""

    type: str
    fields: list[NodeFieldSpec]


class NodeTypeSpec(BaseModel):
    """Everything the builder UI needs to render one node type: form fields + payload types."""

    type: str
    runs_as: Literal["thread", "process"]
    description: str
    accepts: list[str]
    emits: list[str]
    fields: list[NodeFieldSpec]
    # Present only for nodes with a `model_config` field (the detection/classification
    # nodes); None for everything else.
    model_types: list[ModelTypeSpec] | None = None


def _field(
    name: str,
    kind: FieldKind,
    *,
    required: bool = False,
    default: Any = None,
    description: str = "",
    options: list[str] | None = None,
) -> NodeFieldSpec:
    return NodeFieldSpec(
        name=name, kind=kind, required=required, default=default, description=description, options=options
    )


# model_config.type choices shared by ProcessDetInference / ProcessDetBatchInference /
# ProcessBlurInference / ProcessEmbeddingInference. SAHIDetector is excluded (see
# module docstring): its `detector` field needs a live YOLOv8 instance.
_MODEL_TYPES: list[ModelTypeSpec] = [
    ModelTypeSpec(
        type="YOLOv8",
        fields=[
            _field("path", "string", required=True, description="Path to model weights"),
            _field(
                "backend",
                "enum",
                default="TorchBackend",
                options=["TorchBackend", "TensorRTBackend", "ONNXRuntimeBackend"],
                description="Inference backend",
            ),
            _field("device_id", "integer", default=0, description="-1 for CPU"),
            _field("imgsz", "integer", description="Inference image size (square)"),
            _field("conf", "number", default=0.2, description="Confidence threshold"),
            _field("iou", "number", default=0.7, description="NMS IoU threshold"),
            _field("max_det", "integer", default=100, description="Max detections per image"),
            _field(
                "extract_embeddings",
                "boolean",
                default=False,
                description="Also return intermediate-layer embeddings",
            ),
        ],
    ),
    ModelTypeSpec(
        type="BlurClassification",
        fields=[
            _field("path", "string", required=True, description="Path to trained model file"),
            _field(
                "backend",
                "enum",
                default="TorchBackend",
                options=["TorchBackend", "TensorRTBackend", "ONNXRuntimeBackend"],
                description="Inference backend",
            ),
            _field("device_id", "integer", default=0, description="-1 for CPU"),
            _field(
                "conf",
                "number",
                default=0.5,
                description="Ratio-of-blurry-regions threshold above which an image is classified blurry",
            ),
        ],
    ),
    ModelTypeSpec(
        type="EmbeddingFilter",
        fields=[
            _field("path", "string", required=True, description="Path to TorchScript embedding model"),
            _field(
                "backend",
                "enum",
                default="TorchBackend",
                options=["TorchBackend", "TensorRTBackend", "ONNXRuntimeBackend"],
                description="Inference backend",
            ),
            _field("device_id", "integer", default=0, description="-1 for CPU"),
            _field("eps", "number", default=0.015, description="Clustering neighborhood radius"),
            _field("min_samples", "integer", default=2, description="Clustering min samples per cluster"),
            _field("num_extremes", "integer", default=1, description="Extreme (outlier) samples to select per cluster"),
        ],
    ),
]

_MODEL_CONFIG_FIELD = _field(
    "model_config",
    "model_config",  # rendered via the node's top-level `model_types`, not as a plain input
    required=True,
    description="Which model to run and its parameters",
)

_BATCH_FIELDS = [
    _field("batch_size", "integer", required=True, description="Frames per inference batch"),
    _field("batch_queue_size", "integer", default=20, description="Max pending batches queued"),
    _field("batch_collect_timeout", "number", default=0.1, description="Max seconds to wait while filling a batch"),
]

# Static per-node-type field specs. `accepts`/`emits` are filled in dynamically by
# `get_node_catalog()` (see module docstring) — leaving them out here is what keeps
# them from drifting out of sync with `NodeFactory`.
_STATIC_CATALOG: list[dict[str, Any]] = [
    {
        "type": "FolderImageNode",
        "runs_as": "thread",
        "description": "Reads images from a folder",
        "fields": [
            _field("folder_path", "string", required=True, description="Path to the folder of images to read"),
            _field(
                "mode",
                "enum",
                default="loop",
                options=["loop", "only_one"],
                description="loop cycles forever; only_one stops after one pass",
            ),
            _field("frame_delay", "number", default=0.01, description="Seconds to sleep between emitted frames"),
        ],
    },
    {
        "type": "SaveImageNode",
        "runs_as": "thread",
        "description": "Writes frames to disk",
        "fields": [
            _field("save_dir", "string", required=True, description="Directory to write frame_{id}.jpg into"),
        ],
    },
    {
        "type": "ResizeNode",
        "runs_as": "thread",
        "description": "Resizes frames",
        "fields": [
            _field("target_width", "integer", required=True),
            _field("target_height", "integer", required=True),
        ],
    },
    {
        "type": "ResizeProcessNode",
        "runs_as": "process",
        "description": "Resizes frames, isolated in its own process",
        "fields": [
            _field("target_width", "integer", required=True),
            _field("target_height", "integer", required=True),
        ],
    },
    {
        "type": "HashNode",
        "runs_as": "thread",
        "description": "Perceptual-hash dedup filter",
        "fields": [
            _field("delta", "integer", required=True, description="Max Hamming distance to count as a duplicate"),
            _field("hash_size", "integer", required=True),
            _field(
                "hash_type",
                "enum",
                required=True,
                options=["colorhash", "dhash", "phash", "average_hash", "phash_simple", "dhash_vertical", "whash"],
            ),
        ],
    },
    {
        "type": "VadNode",
        "runs_as": "thread",
        "description": "Marks silent AudioChunks as droppable by RMS energy",
        "fields": [
            _field(
                "energy_threshold",
                "number",
                default=0.01,
                description="RMS energy below which an AudioChunk is dropped",
            ),
        ],
    },
    {
        "type": "DrawNode",
        "runs_as": "thread",
        "description": "Draws detected boxes onto the frame",
        "fields": [],
    },
    {
        "type": "TextNormalizeNode",
        "runs_as": "thread",
        "description": "Strips/lowercases TextChunk.text",
        "fields": [],
    },
    {
        "type": "ProcessDetInference",
        "runs_as": "process",
        "description": "Single-frame object detection",
        "fields": [_MODEL_CONFIG_FIELD],
        "model_types": _MODEL_TYPES,
    },
    {
        "type": "ProcessBlurInference",
        "runs_as": "process",
        "description": "Blur detection / filtering",
        "fields": [_MODEL_CONFIG_FIELD],
        "model_types": _MODEL_TYPES,
    },
    {
        "type": "ProcessDetBatchInference",
        "runs_as": "process",
        "description": "Batched object detection",
        "fields": [*_BATCH_FIELDS, _MODEL_CONFIG_FIELD],
        "model_types": _MODEL_TYPES,
    },
    {
        "type": "ProcessEmbeddingInference",
        "runs_as": "process",
        "description": "Batched embedding extraction",
        "fields": [*_BATCH_FIELDS, _MODEL_CONFIG_FIELD],
        "model_types": _MODEL_TYPES,
    },
    {
        "type": "ActiveLearning",
        "runs_as": "thread",
        "description": "Selects frames worth human labelling",
        "fields": [
            _field("num_to_select", "integer", required=True),
            _field("conf_strategy", "enum", required=True, options=["sum", "avg", "max"]),
            _field(
                "conf_weight",
                "number",
                required=True,
                description="Weight of uncertainty vs. embedding distance in the selection score",
            ),
        ],
    },
    {
        "type": "CreateDataset",
        "runs_as": "thread",
        "description": "Assembles the resulting dataset",
        "fields": [
            _field("save_dir", "string", required=True, description="Root directory for images/ and labels/"),
        ],
    },
]


def get_node_catalog() -> list[NodeTypeSpec]:
    """Return the full node-type catalog, with `accepts`/`emits` filled in from `NodeFactory`.

    Returns
    -------
        list[NodeTypeSpec]: One entry per node type the builder UI can add to a graph.

    """
    catalog = []
    for entry in _STATIC_CATALOG:
        node_type = entry["type"]
        accepts = [t.__name__ for t in NodeFactory.get_accepts(node_type)]
        emits = [t.__name__ for t in NodeFactory.get_emits(node_type)]
        catalog.append(NodeTypeSpec(accepts=accepts, emits=emits, **entry))
    return catalog
