# neudc — Neural Dataset Collection

**A streaming pipeline framework for pre-annotating object-detection datasets.**

Point it at a stream of images, wire up a graph of nodes — dedup, detect, embed, select,
save — and get a pre-labelled dataset ready for human review.

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## ✨ What it does

- **Declarative pipelines.** Describe a node graph in YAML; the framework wires the
  transport, spawns the workers and validates the config before anything starts.
- **Nodes run where they should.** Light nodes (readers, resize, filters) run as threads;
  heavy nodes (model inference) run as separate processes, so they get their own GIL and GPU context.
- **Lossless transport.** Nodes talk over ZeroMQ **PUSH/PULL** — messages are queued rather
  than dropped, and a full high-water mark applies backpressure to the producer.
- **Batteries for dataset work.** Perceptual-hash dedup, detection, embeddings,
  active-learning selection and dataset assembly ship as nodes.
- **Observable.** Optional Prometheus metrics endpoint, per-node health monitoring.

## 🧭 How it works

neudc has two planes. The **control plane** turns a YAML file into a running graph:
the config validator fails fast (and names the offending node), the `NodeFactory` lazily
imports only the node types in use, and the `PipelineServiceManager` spawns and wires them.
The **data plane** is the graph itself — each node runs as its own thread or process and
owns a mailbox; producers `PUSH` to each downstream node they declare in `outputs`, so
every edge is an independent 1:1 channel and fan-out is just several edges.

```mermaid
flowchart TB
    subgraph control["🧠 Control plane"]
        direction LR
        yaml["📄 pipeline.yaml"] --> validator["config validator<br/><i>fail-fast · names the bad node</i>"]
        validator --> factory["NodeFactory<br/><i>lazy imports</i>"]
        factory --> manager["PipelineServiceManager<br/><i>start · stop · status</i>"]
    end

    manager -. "spawns and wires" .-> data

    subgraph data["⚙️ Data plane — one running pipeline"]
        direction LR
        reader["FolderImageNode<br/><i>thread</i>"]:::thread
        resize["ResizeProcessNode<br/><i>process</i>"]:::proc
        det["ProcessDetBatchInference<br/><b>process · GPU</b>"]:::gpu
        draw["DrawNode<br/><i>thread · mutates image</i>"]:::thread
        save["SaveImageNode<br/><i>thread</i>"]:::thread
        emb["ProcessEmbeddingInference<br/><b>process · GPU</b>"]:::gpu
        al["ActiveLearning<br/><i>thread</i>"]:::thread
        ds["CreateDataset<br/><i>thread</i>"]:::thread

        reader ==>|Frame| resize
        resize ==>|Batch| det
        det ==>|Frame| draw
        draw ==>|Frame| save
        det -. "fan-out" .-> emb
        emb ==> al
        al ==> ds
    end

    classDef thread fill:#eef6ff,stroke:#3b82f6,color:#1e3a8a;
    classDef proc fill:#fef7ed,stroke:#f59e0b,color:#7c2d12;
    classDef gpu fill:#f0fdf4,stroke:#16a34a,color:#14532d;
```

**Message transport.** A message (`Frame` = image + boxes + metadata, or a `Batch`) is
serialized once per send with pickle protocol 5 and the same bytes feed every edge. Large
buffers can travel *beside* the pickle stream through POSIX shared memory instead of over
TCP — opt-in via `NEUDC_SHM_IMAGES`, and only on single-consumer edges (fan-out keeps
everything in-band to avoid a race on unlink). The consumer copies the buffer into its own
**writable** memory, because nodes like `DrawNode` mutate the image in place.

```mermaid
flowchart LR
    subgraph prod["Producer node"]
        p1["process()"] --> p2["mailbox.send(msg)"]
        p2 --> enc["codec.dumps<br/><i>pickle protocol-5</i>"]
    end

    enc ==>|"envelope: metadata + small buffers"| push(["ZMQ PUSH"])
    enc -. "large out-of-band buffers<br/>image · embedding · tokens" .-> shm[("POSIX shared memory<br/><i>opt-in · single-consumer</i>")]

    push ==>|"TCP · 1:1 · backpressure"| pull(["ZMQ PULL"])

    subgraph cons["Consumer node"]
        pull --> rx["rx-thread<br/><i>bounded queue</i>"]
        rx --> dec["codec.loads"]
        dec --> out["process()<br/><b>writable Frame</b>"]
    end

    shm -. "open → copy → unlink" .-> dec

    classDef io fill:#f1f5f9,stroke:#64748b,color:#0f172a;
    class push,pull io;
```

A rendered architecture diagram lives in [`assets/schema/neudc.pdf`](assets/schema/neudc.pdf).

## 🚀 Quick start

```bash
git clone https://github.com/Bleaff/data_pipeline_orchestration.git
cd data_pipeline_orchestration
python3 -m venv env && source env/bin/activate
python3 -m pip install -e .
```

Run the bundled CPU-only example (no GPU, no model required — put some images in
`assets/images/` first):

```bash
python3 -m neudc.entrypoints.main assets/configs/example_cpu_pipeline.yaml
```

Run several pipelines side by side, each in its own process:

```bash
python3 -m neudc.entrypoints.main_multipipe assets/configs/multipipe.yaml
```

## ⚙️ Configuration

A pipeline is a list of nodes. Each node has an `id`, a `type`, its own parameters, and
`outputs` — the ids it forwards frames to.

```yaml
nodes:
  - id: reader
    type: FolderImageNode
    folder_path: "assets/images"
    mode: "loop"            # "loop" cycles forever, "only_one" stops after one pass
    frame_delay: 0.1
    outputs: [resize]

  - id: resize
    type: ResizeNode
    target_width: 640
    target_height: 480
    outputs: [saver]

  - id: saver
    type: SaveImageNode
    save_dir: "./output_images"
    outputs: []
```

The config is validated at startup, so mistakes fail fast and legibly: an unknown node
`type`, a duplicate `id`, or an `outputs` entry that points at no node are all rejected
before a single frame moves.

Optional metrics endpoint:

```yaml
prometheus:
  enable: true
  port: 8000
```

Every node exposes a common set of Prometheus metrics, all labeled by `node`:

| Metric | Type | Meaning |
|---|---|---|
| `neudc_node_queue_depth` | Gauge | Current depth of the node's internal message queue |
| `neudc_node_queue_hwm` | Gauge | Configured ZMQ high-water mark for the node's mailbox |
| `neudc_node_messages_processed_total` | Counter | Messages `process()` completed successfully |
| `neudc_node_process_latency_seconds` | Histogram | `process()` wall-clock latency per attempt |
| `neudc_node_errors_total` | Counter | `process()` exceptions raised (including retried attempts) |
| `neudc_node_retries_total` | Counter | Retry attempts under an `on_error: retry` policy |
| `neudc_node_dropped_total` | Counter | Messages dropped after the error policy gave up |
| `neudc_node_dead_lettered_total` | Counter | Messages written to a node's dead-letter sink |
| `neudc_node_failures_total` | Counter | Times a node gave up under an `on_error: fail` policy |
| `neudc_node_health` | Gauge | Node health as observed by its own run loop (1=healthy, 0=unhealthy) |

A ready-made dashboard for these is at
[`docs/grafana/neudc-dashboard.json`](docs/grafana/neudc-dashboard.json) — import it
into Grafana against your Prometheus datasource.

**Process nodes and multiprocess mode.** Inference nodes run in a separate OS process
(`BaseProcessNode`), so by default their metrics live in that process's own private
registry — invisible to the endpoint above, which only serves the main process. To make
process-node metrics visible too, set `PROMETHEUS_MULTIPROC_DIR` to a writable, empty
directory *before starting the pipeline* (this is a hard requirement of
[`prometheus_client`'s multiprocess mode](https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn),
not a neudc-specific setting — the env var must exist before `prometheus_client` is first
imported anywhere in the process tree):

```bash
export PROMETHEUS_MULTIPROC_DIR=/tmp/neudc-prometheus
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
python -m neudc.entrypoints.main config/pipeline.yaml
```

Without it, only same-process (`BaseThreadedNode`) metrics are visible — the historical,
single-process behaviour.

### Error policy

Any node may declare what happens when its processing raises. Without the block a
node logs the exception and drops the message — the historical behaviour.

```yaml
  - id: detector
    type: ProcessDetInference
    outputs: [saver]
    error_policy:
      on_error: retry          # skip (default) | retry | fail
      max_retries: 3
      initial_backoff: 0.1     # seconds before the first retry
      backoff_multiplier: 2.0  # each retry waits this much longer
      max_backoff: 5.0         # …up to this cap
      jitter: true             # spread delays so nodes don't retry in lockstep
      dead_letter_dir: "./dead_letter"
      store_payload: false     # also pickle the message next to the record
```

| `on_error` | Behaviour |
|---|---|
| `skip` | Log, dead-letter, drop the message, keep going. The default. |
| `retry` | Re-run `process()` up to `max_retries` times with exponential backoff, then dead-letter and drop. Requires `max_retries >= 1`. |
| `fail` | Dead-letter and stop the node, which brings the whole pipeline down. |

With `dead_letter_dir` set, every message that could not be processed is appended as
one JSON object to `<dir>/<node_id>.jsonl` — the error, its traceback, and whatever
identifies the message. Payloads are not written unless `store_payload` is on, since a
dead-lettered frame would otherwise cost megabytes per record.

Two things to know before choosing `retry`:

- Only `process()` is retried, never the send that follows it — re-sending would
  deliver the message twice.
- `process()` must tolerate a second call on the same message. Nodes that mutate it in
  place (`DrawNode` draws onto `frame.image`) would retry on a half-modified message;
  use `skip` for those.

The block is validated with the rest of the config, so a typo or a contradictory
setting (`on_error: retry` with no retries) is rejected at startup, naming the node.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NEUDC_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `NEUDC_LOG_FILE` | *(unset)* | Path to a rotating log file. Opt-in — off by default to keep the hot path free of disk I/O. |
| `NEUDC_SHM_IMAGES` | `0` | Opt-in POSIX shared-memory transport for large buffers (images, embeddings, …) instead of pickling them over TCP. Applies only to **single-consumer** edges; fan-out and control messages stay in-band. POSIX only. |
| `NEUDC_SHM_MIN_BYTES` | `65536` | Minimum buffer size (bytes) diverted to shared memory when `NEUDC_SHM_IMAGES` is on; smaller buffers stay in-band. |

## 🧩 Available nodes

| Node | Runs as | What it does |
|---|---|---|
| `FolderImageNode` | thread | Reads images from a folder (`folder_path`, `mode`, `frame_delay`) |
| `ResizeNode` | thread | Resizes frames (`target_width`, `target_height`) |
| `ResizeProcessNode` | process | Same, isolated in its own process |
| `HashNode` | thread | Perceptual-hash dedup filter (`delta`, `hash_size`, `hash_type`) |
| `DrawNode` | thread | Draws detected boxes onto the frame |
| `SaveImageNode` | thread | Writes frames to disk (`save_dir`) |
| `ProcessDetInference` | process | Single-frame object detection (`model_config`) |
| `ProcessDetBatchInference` | process | Batched object detection (`batch_size`, `model_config`) |
| `ProcessBlurInference` | process | Blur detection / filtering |
| `ProcessEmbeddingInference` | process | Batched embedding extraction |
| `ActiveLearning` | thread | Selects frames worth human labelling |
| `CreateDataset` | thread | Assembles the resulting dataset |
| `TextNormalizeNode` | thread | Non-CV: strips/lowercases `TextChunk.text` (`accepts`/`emits` = `TextChunk`) |

Model-backed nodes take a `model_config` naming the backend (`TorchBackend`,
`ONNXBackend`, `TRTBackend`), the weights `path` and the `device_id` (`-1` for CPU).

## 🛠 Development

```bash
python3 -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Run the tests:

```bash
python3 -m pytest test
```

Optional extras: `pip install -e ".[dev]"`, `".[trt]"` (TensorRT), `".[onnx]"` (ONNX Runtime).

## 🗺 Roadmap

The full roadmap lives in [docs/ROADMAP.md](docs/ROADMAP.md); task status is tracked on the
[project board](https://github.com/users/Bleaff/projects/6). Highlights:
zero-shot / open-vocabulary labelling, video support with tracking and label propagation,
a control-plane API with a dashboard, shared-memory image transport, and diffusion-based
image generation (local models or hosted API endpoints).

## 📄 License

Copyright (C) 2025–2026 Sergey Sysoev

This project is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.

Note that the AGPL requires anyone who runs a modified version of this software as a
network service to make the corresponding source code available to its users.
