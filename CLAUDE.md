# CLAUDE.md

Guidance for any coding agent (and human) working in **neudc** — a data-pipeline
orchestration framework for auto-labelling (detection / embeddings / active learning).
Nodes → pipelines (separate processes) → ZMQ mailboxes, wired from YAML via a factory.

This file exists to keep development from rotting. **Read it before writing code.**
The three non-negotiables are: **tests**, **verification**, **clean, well-placed logic**.

---

## Golden rules (do not skip)

1. **Every behavioural change ships with a test.** New node, new backend, new codec,
   bugfix — it gets a test under `test/` mirroring the source path. No test → not done.
2. **Verify before you claim done.** Run the full suite *and* the linters locally, and
   actually exercise the path you changed (see [Definition of done](#definition-of-done)).
   If something failed or was skipped, say so — never report green when it isn't.
3. **Put logic where it belongs.** Match the existing layering (below). Don't inline
   behaviour into a base class, an entrypoint, or a factory to "make it work".
4. **Leave the repo cleaner than you found it.** No scratch files, no logs, no
   commented-out code, no debug prints committed.

---

## Commands

```bash
# Tests — must pass before every commit
python3 -m pytest test -q

# Single file / test while iterating
python3 -m pytest test/core/communication/test_zeroqueue.py -q

# Lint / format — same checks CI runs (black, isort, pyupgrade, ruff, codespell…)
pre-commit run --all-files

# Editable install (+ optional extras)
pip install -e .
pip install -e ".[dev]"   # dev tooling
pip install -e ".[trt]"   # TensorRT backend
pip install -e ".[onnx]"  # ONNX Runtime backend
```

Python floor is **3.11** (code uses `enum.StrEnum` / `typing.Self`). Line length is **120**.

---

## Architecture — where code goes

Respect these boundaries. Logic that crosses them is a smell.

| Layer | Path | Responsibility |
|---|---|---|
| Abstractions | `neudc/core/base/` | `BaseNode`, `BaseProcessNode`, `BaseThreadedNode`, `BasePipeline`, queues, mailbox contracts. **Interfaces only** — no concrete behaviour. |
| Node implementations | `neudc/core/node/` | Concrete nodes: `readers/`, `processors/`, `model/`, `filters/`, `broadcast/`. Behaviour lives in **mixins**; the node class only wires config → mixin. |
| Communication | `neudc/core/communication/` | ZMQ transport: `mailbox/`, `zero_queue/`, `messaging/`, codecs. Hot-path code — keep allocation-light and covered by hotpath tests. |
| Models / backends | `neudc/nn/` | `models/` (det, cls) and `backends/` (torch, onnxruntime, trt). Backends are pluggable behind `backends/base.py`. |
| Config | `neudc/schemas.py`, `neudc/core/utils/` | Pydantic schemas + loader/validator. Validation fails fast and names the offending node. |
| Service | `neudc/service/` | `PipelineServiceManager` — start/stop/status of pipelines. |
| Entrypoints | `neudc/entrypoints/` | Thin wiring only. No business logic here. |

**A node type only runs if registered** in `NodeFactory.NODE_IMPORTS`
(`neudc/core/node/node_factory.py`) and structurally accepted by the config validator
(`neudc/core/utils/config_schema.py`). Imports there are lazy — don't add heavy
top-level imports that break CPU-only pipelines.

---

## Adding a node (the common task)

Full walkthrough: [docs/NODE_DEV_GUIDE.md](docs/NODE_DEV_GUIDE.md). In short:

1. **Logic in a mixin** — e.g. `core/node/processors/mixins/<name>_mixin.py`, implementing `process()`.
2. **Node class** = mixin + base (`BaseThreadedNode` or `BaseProcessNode`), with `from_config()`.
3. **Register** in `NodeFactory.NODE_IMPORTS` (module path + class name, lazy).
4. **Test** the mixin's `process()` in isolation *and* the node's `from_config()` under `test/`.
5. **Document** any new config keys where users will look (README / node guide).

Keep `process()` pure and testable: data in → data out, no hidden global state.

---

## Code style

- Type hints on public functions; `from __future__ import annotations` at the top of modules.
- Docstrings on public classes/methods (the codebase does this consistently — match it).
- **Code, comments, and docstrings in English.** Narrative docs (`docs/*.md`) may be Russian.
- Config via **Pydantic** models (`ConfigDict`, not the deprecated class-based `Config`).
- Logging via `neudc.utils.LOGGER` — **never `print`**. Catch narrowly; log with context.
- **No mutable default arguments** (`def f(x: dict = {})`) — use `None` and assign inside.

---

## Testing requirements

- Tests live under `test/`, mirroring the source tree (`neudc/core/node/... → test/core/node/...`).
- Cover the **happy path and at least one failure/edge case** (bad config, empty input, timeout).
- For communication/hot-path changes, add or update the `*_hotpath.py` tests.
- Inference nodes with heavy model deps: test config parsing, wiring, and pre/post-processing
  with lightweight fakes/mocks — don't require GPU or real weights in CI.
- The suite must stay green and hermetic (no network, no real model downloads).

---

## Definition of done

Before you say a change is complete, all of these are true:

- [ ] `python3 -m pytest test -q` passes locally.
- [ ] `pre-commit run --all-files` is clean.
- [ ] New/changed behaviour has a test that would fail without the change.
- [ ] You exercised the actual path (ran the node/pipeline or the new code), not just tests.
- [ ] New node types are registered in the factory **and** accepted by the config validator.
- [ ] No stray files, logs, prints, or commented-out blocks added.
- [ ] Public config keys / behaviour changes are documented.

---

## Don't (things that make the codebase rot)

- Don't commit scratch/experiment files, `output.log`, notebooks, or `sandbox/` throwaways.
- Don't add a node without registering it — it will silently never run.
- Don't put behaviour in `core/base/*` or `entrypoints/*`.
- Don't bypass the config schema by reading raw dicts deep in the call stack.
- Don't widen a `try/except` to hide a failure; fix the cause and let it surface.
- Don't report "done" with failing/skipped tests or unrun verification.
```
