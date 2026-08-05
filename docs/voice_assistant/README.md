# Voice assistant: mic → VAD → ASR → LLM → TTS → speaker

An end-to-end voice assistant built entirely from `neudc` nodes, running as a single
local pipeline. Heavy inference (ASR, LLM, TTS) happens on a remote GPU box, reached
purely over HTTP — no ZMQ cross-machine transport is used or needed.

```
                         local machine (one neudc pipeline, one process)
 ┌───────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────────┐
 │AudioReader│──▶│ VadNode │──▶│ AsrNode │──▶│ LlmNode │──▶│ TtsNode │──▶│AudioPlayerNode│
 │  (mic)    │   │(silence │   │(HTTP →  │   │(HTTP →  │   │(HTTP →  │   │  (speaker)   │
 │           │   │ tagging)│   │ remote) │   │ remote) │   │ remote) │   │              │
 └───────────┘   └─────────┘   └────┬────┘   └────┬────┘   └────┬────┘   └──────────────┘
                                     │             │             │
                                     ▼             ▼             ▼
                          ┌──────────────────────────────────────────────┐
                          │        remote GPU box (bleaf@…, 4060ti)       │
                          │  faster-whisper-server │ llama-server+Gemma │ openedai-speech │
                          │  /v1/audio/transcriptions │ /v1/chat/… │ /v1/audio/speech
                          └──────────────────────────────────────────────┘
```

**Deployed and verified 2026-08-04** on `bleaf@192.168.1.153`: all three services are up
via `docker compose -f docker-compose.remote.yml up -d` and were round-trip tested —
`TtsNode`'s synthesized speech fed back into `AsrNode` transcribed to the exact input
text, and the LLM answered a real chat-completion request correctly. See
[Deployment notes](#deployment-notes-from-the-first-real-rollout) below for what that
rollout actually required (it wasn't just `docker compose up`).

`VadNode`'s existing `chunk.drop` flag doubles as utterance segmentation: `AsrNode`
buffers speech chunks and flushes one transcription call per utterance (not per
0.5s chunk) as soon as a silent chunk follows. `TtsNode` mirrors that on the way
back — it buffers `LlmNode`'s output (streamed deltas or one final reply, either
works) and synthesizes once the reply is complete.

All three remote calls speak the OpenAI wire format, the same convention the
already-existing `LocalLLMBackend` (`neudc/nn/backends/llm/local.py`) uses — so every
new backend here is just "point `base_url` at the right self-hosted server."

## 1. Deploy the remote services

On `bleaf@192.168.1.153` (the 4060ti box), all three run as containers from the same
`docker-compose.remote.yml`:

| Service | Port | What it is |
|---|---|---|
| ASR (`faster-whisper-server`) | `5241` | `Systran/faster-whisper-large-v3`, GPU |
| LLM (`llama-server` + Gemma) | `5242` | the host's own CUDA-built `llama.cpp` binary, containerized — see notes below |
| TTS (`openedai-speech`) | `5243` | Piper voices behind an OpenAI-compatible `/v1/audio/speech` |

```bash
scp docker-compose.remote.yml bleaf@192.168.1.153:~/voice-assistant/
ssh bleaf@192.168.1.153
cd ~/voice-assistant
docker compose -f docker-compose.remote.yml up -d
```

Healthchecks (run from either machine, adjust the host if not local):

```bash
# ASR
curl -s http://192.168.1.153:5241/v1/models

# LLM
curl -s http://192.168.1.153:5242/v1/models

# TTS
curl -s http://192.168.1.153:5243/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","voice":"alloy","input":"hello","response_format":"wav"}' \
  --output /tmp/test.wav
```

**Sample rate**: the mic/ASR side is 16 kHz mono (`faster-whisper-server` resamples
internally regardless of input rate, confirmed by feeding it 22.05 kHz TTS output
directly). The TTS side is **not** 16 kHz — the deployed Piper voices emit **22050 Hz**
mono, and that isn't user-configurable per request. `AudioChunk.sample_rate` already
travels with each message, so no pipeline code needs to change — just open the
playback `SoundDeviceAudioOutputDevice` at 22050 Hz, not 16000 (see the runner script
below).

## 2. Point the pipeline config at your services

`assets/configs/voice_assistant_pipeline.yaml` wires the full graph, already pointed at
the deployed services above. Update `asr_config.base_url` / `llm_config.base_url` /
`tts_config.base_url` if your ports differ, and `llm_config.model_id` /
`tts_config.voice` to match whatever's actually being served (`llama-server` and
`openedai-speech` both accept an arbitrary/alias model name here — confirmed by
testing — so this field is mostly cosmetic, not strictly validated).

New per-node config keys:

| Node | Key | Meaning |
|---|---|---|
| `AsrNode` | `asr_config.model_id` / `base_url` / `api_key` / `timeout` | ASR server identity and endpoint |
| `AsrNode` | `session_id` | Stamped onto every transcribed `TextChunk` |
| `AsrNode` | `min_speech_duration` | Buffered speech shorter than this (seconds) is discarded as noise |
| `LlmNode` | `llm_config.*` | A `LLMBackendConfig` (`provider`, `model_id`, `base_url`, `stream`, `temperature`, `max_tokens`, `timeout`) — same schema the (previously unwired) `LLMBackend` layer already defined |
| `LlmNode` | `system_prompt` | Optional system message seeding the conversation history |
| `TtsNode` | `tts_config.model_id` / `voice` / `base_url` / `api_key` / `timeout` | TTS server identity, voice, and endpoint |

## 3. Run it locally

Real microphone/speaker access needs live Python objects that plain YAML can't
express — the same reason `AudioReaderNode.from_config` already requires an injected
`device` (see `assets/configs/audio_vad_pipeline.yaml`). `AudioPlayerNode` follows the
identical pattern for output. Install the optional `audio` extra, then run a small
script that mirrors `neudc.entrypoints.main.main()` but injects both devices before
building the nodes:

```bash
pip install -e ".[audio]"
```

```python
from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.broadcast.audio_player import SoundDeviceAudioOutputDevice
from neudc.core.node.node_factory import NodeFactory
from neudc.core.node.readers.audio_reader import SoundDeviceAudioDevice
from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import validate_pipeline_config

config = load_config("assets/configs/voice_assistant_pipeline.yaml")
validate_pipeline_config(config)

for node_config in config["nodes"]:
    if node_config["id"] == "reader":
        node_config["device"] = SoundDeviceAudioDevice(sample_rate=16000, channels=1)
    elif node_config["id"] == "player":
        # 22050 Hz to match the deployed Piper voices' native output rate, not 16000.
        node_config["device"] = SoundDeviceAudioOutputDevice(sample_rate=22050, channels=1)

router = RoutingFactory(config)
mailbox_map = router.create_mailboxes()
nodes = [
    NodeFactory.create(node_config, mailbox=mailbox_map[node_config["id"]][0])
    for node_config in config["nodes"]
]
for node in nodes:
    node.start()

input("Voice assistant running — press Enter to stop.\n")
for node in nodes:
    node.stop()
```

Speak into the mic; once VAD detects silence after speech, you should hear a spoken
reply within a few seconds (network + inference latency on the remote box).

## 4. Watching it: metrics and the dashboard

Start the [control-plane API](../../README.md#-control-plane-api) (`neudc.service.api`)
**in the same process** as the snippet above, right after building `nodes` and before
`node.start()` — it must share that process's Prometheus registry to have anything to
show, and a separately-run `python -m neudc.entrypoints.api_server` would have none of
this pipeline's metrics in it:

```python
import uvicorn
from neudc.service.api import create_app

app = create_app()
uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
server = uvicorn.Server(uvicorn_config)
threading.Thread(target=server.run, daemon=True).start()
```

`GET /metrics` (raw Prometheus text, includes the `neudc_node_process_latency_seconds`
histogram) and `GET /ws/metrics` (a JSON snapshot pushed once a second) now reflect
`reader`/`vad`/`asr`/`llm`/`tts`/`player`'s real throughput/latency/errors as they run.

For the graphical dashboard (`frontend/`, `npm run dev`) to show this pipeline at
`/pipelines/voice-assistant` too, register it — `PipelineServiceManager` normally
only knows about pipelines it started itself via `/start` (a JSON-only contract this
pipeline's live mic/speaker `device` objects can't satisfy, same reason as above), so
without this the dashboard's pipeline list is just empty:

```python
import requests

requests.post(
    "http://127.0.0.1:8000/pipelines/voice-assistant/register",
    json={"nodes": load_config("assets/configs/voice_assistant_pipeline.yaml")["nodes"]},
)
```

This only makes the pipeline *visible* — the manager never starts, stops, or
health-checks it (see `PipelineServiceManager.register_external_pipeline`); reports it
as `"running"` for as long as it stays registered. Call the mirror
`POST /pipelines/voice-assistant/unregister` when the script exits (`node.stop()`
doesn't do this automatically).

`run_voice_assistant_scratch.py`-style runner scripts typically do all of this by
default (API on `:8000`, auto-registered, `--no-api` to skip it) — see the script's
own `--help` if you have one from an earlier session.

## Deployment notes from the first real rollout

What the plan above assumed vs. what was actually true on `bleaf@192.168.1.153`,
worth knowing before touching this box again:

- **No vLLM was ever installed.** `ollama` is present but was found crash-looping
  (`mkdir .../models/blobs: permission denied`, 2600+ restarts) — a pre-existing,
  unrelated problem, left as-is. The LLM service instead runs the host's own
  CUDA-built `~/Desktop/Projects/llama.cpp/build/bin/llama-server` against
  `~/.ollama/models/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-Q4_K_XL.gguf`, wrapped in a
  container (see `docker-compose.remote.yml`'s `llm` service and its comments) rather
  than reinstalled/rebuilt inside Docker.
- **That binary needs its own container recipe.** It's dynamically linked against a
  newer glibc/libgomp/libstdc++ than stock CUDA runtime images ship (Manjaro host,
  rolling release). Bind-mounting it into `nvidia/cuda:...-ubuntu24.04` and invoking
  it through the container's own linker fails (`libgomp.so.1: cannot open shared
  object file`). The compose file works around this by also bind-mounting the host's
  `/usr/lib` and invoking the binary through *that* `ld-linux` as an explicit
  interpreter, so it always runs against the userspace it was actually built for.
- **This Gemma checkpoint "thinks" by default.** Without `--reasoning-budget 0`, it
  spends completion tokens on a hidden `reasoning_content` chain-of-thought before
  answering — 1.9s/63 tokens for a one-sentence answer, and at a low `max_tokens` the
  budget can be exhausted entirely on thinking with an **empty** `message.content`.
  With `--reasoning-budget 0`: ~300ms, direct answer, no reasoning field. Already set
  in the compose file.
- **`--gpus all` was broken for every container on the host**, not just these three:
  a generated CDI manifest (`/etc/cdi/nvidia.yaml`) had gone stale after a driver
  update and pointed at a `libnvidia-egl-gbm.so` file version that no longer existed.
  Fixed with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` (regenerates
  a derived config file from the currently installed driver; no data touched). If GPU
  containers start failing again after a future driver update on this host, this is
  the first thing to check.
- **TTS voices are 22050 Hz, not 16 kHz** — see the Sample rate note above.

## Out of scope (documented, not built)

- **Barge-in / cancellation** while `TtsNode` is playing a reply. The primitives
  already exist (`ControlMessage`, `control_outputs` in the config schema) but no
  node currently produces a cancel signal on renewed speech — that would mean
  extending `VadNode` to detect "speech resumed while a reply is in flight" and send
  a `ControlMessage(action=CANCEL)` to a `control_outputs` edge.
- **Resampling** if the deployed TTS voice can't emit 16 kHz natively.
