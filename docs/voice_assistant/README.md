# Voice assistant: mic → VAD → ASR → LLM → tools → TTS → speaker

An end-to-end voice assistant built entirely from `neudc` nodes, running as a single
local pipeline. Heavy inference (ASR, LLM, TTS) happens on a remote GPU box, reached
purely over HTTP — no ZMQ cross-machine transport is used or needed.

```
                    local machine (one neudc pipeline, one process)
 ┌───────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐
 │AudioReader│─▶│ VadNode │─▶│ AsrNode │─▶│ LlmNode │─▶│AgentTool │─▶│ TtsNode │─▶│AudioPlayerNode│
 │  (mic)    │  │(silence │  │(HTTP →  │  │(HTTP →  │  │  Node    │  │(HTTP →  │  │  (speaker)   │
 │           │  │ tagging)│  │ remote) │  │ remote) │  │ (local)  │  │ remote) │  │              │
 └───────────┘  └────▲────┘  └────┬────┘  └────┬────┘  └──────────┘  └────┬────┘  └──────┬───────┘
                      │            │            │                          │              │
                      └────────────┼────────────┼──────── AEC ref ─────────┼──────────────┘
                                    ▼            ▼                          ▼
                          ┌──────────────────────────────────────────────┐
                          │        remote GPU box (bleaf@…, 4060ti)       │
                          │  faster-whisper-server │ llama-server+Gemma │ openedai-speech │
                          │  /v1/audio/transcriptions │ /v1/chat/… │ /v1/audio/speech
                          └──────────────────────────────────────────────┘
```
`vad`'s `control_outputs` also fan out a priority cancel edge straight to
`llm`/`tool`/`tts`/`player` (not shown above) — see
[Barge-in](#barge-in-interrupting-a-reply-by-talking-over-it). `AgentToolNode` runs
entirely locally (no remote HTTP call of its own, aside from whatever `fetch_url`
itself is asked to fetch) — see [Agent tools](#agent-tools) below.

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
| `VadNode` | `session_id` | Stamped onto every `AudioChunk`/cancellation this node emits — it owns turn-boundary assignment for the whole downstream chain (see Barge-in below) |
| `VadNode` | `control_outputs` | Nodes that get a priority CANCEL on barge-in (`[llm, tool, tts, player]` in the shipped config) |
| `VadNode` | `aec_enabled` / `aec_filter_length` / `aec_step_size` / `mic_sample_rate` | Acoustic echo cancellation — see [Echo cancellation](#echo-cancellation-aec) below |
| `AudioPlayerNode` | `outputs` | Wire back to `vad` (`[vad]`) so it receives AEC reference audio — see below |
| `AsrNode` | `asr_config.model_id` / `base_url` / `language` / `api_key` / `timeout` | ASR server identity, endpoint, and an optional language hint (e.g. `"ru"`) pinning transcription instead of relying on per-request auto-detection |
| `AsrNode` | `asr_config.no_speech_threshold` | Default `0.6` — rejects a transcription (as if it were empty) when Whisper's own `no_speech_prob` says the audio probably wasn't real speech; `null` disables the check — see [ASR hallucination on silence/noise](#asr-hallucination-on-silencenoise) |
| `AsrNode` | `min_speech_duration` | Buffered speech shorter than this (seconds) is discarded as noise |
| `LlmNode` | `llm_config.*` | A `LLMBackendConfig` (`provider`, `model_id`, `base_url`, `stream`, `temperature`, `max_tokens`, `timeout`) — same schema the (previously unwired) `LLMBackend` layer already defined |
| `LlmNode` | `system_prompt` | Optional system message seeding the conversation history — include the tool-call syntax here (see [Agent tools](#agent-tools)) if `tool` is wired in |
| `AgentToolNode` | `enable_timers` / `enable_web_fetch` / `enable_file_io` | Which tool groups are active (all default `true`) |
| `AgentToolNode` | `sandbox_dir` | Root directory `create_file`/`read_file` are confined to (required if `enable_file_io`) |
| `AgentToolNode` | `session_id` | Stamped on a timer's reminder `TextChunk` when it later fires — set to match `vad.session_id` |
| `AgentToolNode` | `fetch_timeout` / `fetch_max_chars` | `fetch_url`'s HTTP timeout and extracted-text truncation length |
| `TtsNode` | `tts_config.model_id` / `voice` / `speed` / `base_url` / `api_key` / `timeout` | TTS server identity, voice, playback speed (`1.0` = normal, `openedai-speech`'s own range is `0.25`-`4.0`; omit for provider default), and endpoint |
| `AudioPlayerNode` | `playback_check_interval` | Seconds of audio per block between barge-in cancellation checks (default 0.1) |

## Barge-in (interrupting a reply by talking over it)

`VadNode` detects a rising edge (silence → speech) as "the user started talking
again." If a previous turn was still active, that's a barge-in: it broadcasts a
priority `ControlMessage(action=CANCEL)` for the previous turn over the control
channel (`control_outputs`), which reaches `llm`/`tts`/`player` out-of-band —
independent of however backed-up their data queues are.

What that actually stops, concretely:
- **`LlmNode`** (with `llm_config.stream: true`, the shipped default): generation is
  checked for cancellation *between tokens*, so word generation genuinely stops mid-reply.
  With `stream: false` it can't — a single blocking HTTP call has no interruption
  point once fired.
- **`TtsNode`**: any not-yet-processed chunk for a cancelled turn is skipped outright
  (`BaseNode._handle` checks before calling `process()`); a `synthesize()` call
  already in flight for that turn still completes, but its audio is never played
  (see next point) — a little wasted GPU time, not an audible bug.
- **`AudioPlayerNode`**: plays each `AudioChunk` in `playback_check_interval`-sized
  blocks, re-checking cancellation between blocks — this is what actually cuts
  off audio already coming out of the speaker, not just future chunks that hadn't
  started playing yet.
- **`AgentToolNode`**: a not-yet-processed chunk for a cancelled turn is skipped
  outright, same as `TtsNode` above — but a tool call already in flight (e.g. a slow
  `fetch_url`) still runs to completion, since `process()` is a single synchronous
  call with no mid-call interruption point (unlike `LlmNode`'s streaming generator or
  `AudioPlayerNode`'s per-block checks). `fetch_timeout` bounds how long that can be.

Known limitation: `VadNode` can't distinguish real speech from a loud noise (a cough,
a door) — it's the same RMS-energy heuristic as before, just now driving turn
boundaries too. A loud blip can trigger a spurious cancel of an in-flight reply, same
false-positive rate as the existing VAD always had. A real VAD model (e.g. Silero)
would reduce this; out of scope here.

## Agent tools

`AgentToolNode` sits between `llm` and `tts` (`LlmNode -> AgentToolNode -> TtsNode`)
and resolves inline tool-call markers the LLM writes in its reply text — timers/
reminders, web page fetch+parse, and sandboxed file create/read, the three tools
enabled by default (`enable_timers`/`enable_web_fetch`/`enable_file_io`).

**Protocol**: the LLM writes `[[tool_name({"arg": "value"})]]` anywhere in its reply
(arguments are a flat JSON object); `AgentToolMixin` finds every marker via regex,
executes the named tool synchronously, and replaces the marker in place with the
tool's returned text — or a short `[tool error: ...]` message if the name is unknown,
the JSON doesn't parse, or the tool itself failed (e.g. an unreachable URL, a path
outside the sandbox). The LLM has to actually be told this syntax exists —
`llm.system_prompt` in the shipped config includes the exact marker format and the
tool list (kept in sync by hand with what `tool_system_prompt()` in
`agent_tool_mixin.py` would generate for the same tool set).

**This is a single pass, not a multi-turn agent loop**: a tool's result is spliced
directly into the reply the LLM already generated — it is never fed back to `LlmNode`
for a second generation call. That keeps the wiring a straight line with no cycle
back into `llm`, at the cost of the LLM never seeing a tool's actual result before it
finishes composing the sentence around the call. Fine for confirmation-style tools
(a timer was set, a file was saved) where the LLM doesn't need the result to phrase a
sensible reply; weaker for a tool whose result should shape the wording, e.g.
summarizing a fetched page in the LLM's own words rather than the page's. A second
pass through `LlmNode` — feeding the resolved text back in as a follow-up turn — is a
natural extension, not built here.

**The three tools**:
- `set_timer({"seconds": <number>, "label": <string, optional>})` — returns an
  immediate confirmation (never blocks `process()` for the timer's duration); the
  actual reminder is delivered later as a new `TextChunk` sent straight through this
  node's own mailbox (`source="timer"`, `session_id` from config, `is_final=True`),
  so it flows through `tts`/`player` and gets spoken on its own once the timer fires.
  Bounded to at most 24h (`MAX_TIMER_SECONDS`) as a guard against a typo'd/runaway
  timer, not a real product limit.
- `fetch_url({"url": <string>})` — `http(s)`-only, with an HTTP timeout
  (`fetch_timeout`), a response-size cap, and visible-text extraction via the stdlib
  `html.parser` (no new dependency) rather than returning raw HTML; extracted text is
  truncated to `fetch_max_chars`. These caps are a page-size guard, not a security
  boundary — the LLM is the only thing that ever calls this tool.
- `create_file({"path": <string>, "content": <string>})` / `read_file({"path": <string>})`
  — confined to `sandbox_dir`: every path is resolved (`Path.resolve()`) and checked
  against the sandbox root (`Path.is_relative_to`) before any filesystem access, so
  neither `../` traversal nor an absolute path can reach outside it. Content/read
  length are capped (`MAX_FILE_CONTENT_CHARS`/`MAX_READ_CHARS` in `file_tools.py`).

**Autonomous, not confirmation-gated**: all three tools execute as soon as their
marker is parsed — there is no "ask before running" step, including for the
side-effecting ones (files, network). The sandbox directory is the safety boundary
for file I/O; there is currently no equivalent boundary narrowing what `fetch_url`
can reach (any `http(s)` URL) beyond the caps above.

**Not exercised against a live LLM** as part of this change — the marker-parsing
protocol, each tool, and `AgentToolNode.from_config` are covered by hermetic tests
(`test/core/node/processors/mixins/`), but whether Gemma reliably follows the
`[[tool_name({...})]]` syntax in practice (vs., say, describing the JSON in prose
instead of emitting the literal marker) has not been verified against the real
deployed service.

## Echo cancellation (AEC)

On a laptop with the mic and speakers close together, the mic picks up the
assistant's own TTS output as "the user talking again," and barge-in fires on
itself. `assets/configs/voice_assistant_pipeline.yaml` wires `player.outputs: [vad]`
so `AudioPlayerNode` feeds a copy of every block it actually plays back to `vad`
(tagged `source="speaker-reference"`), and `vad.aec_enabled: true` turns on an
adaptive NLMS filter (`neudc/core/node/filters/mixins/aec.py`) that estimates and
subtracts that echo from mic chunks *before* the RMS/turn logic runs — both silence
detection and what `AsrNode` transcribes are working from the cleaned signal, not the
raw one.

This is a software estimate, not physical isolation — it adapts continuously and
needs a little time (a few hundred ms of actual playback) to converge on a given
room/speaker/mic setup, so the very first moment of a reply is less well cancelled
than the rest. `aec_filter_length` (taps, at `mic_sample_rate`) bounds the
acoustic+buffering delay it can compensate for — 2048 samples at 16 kHz is ~128ms;
raise it if echo is still leaking through after convergence and the room has more
delay/reverb than that. `aec_step_size` is the adaptation rate (mu) — higher adapts
faster but is less stable. A double-talk guard inside the filter freezes adaptation
(without giving up cancellation of the *current* echo estimate) whenever the mic
looks dominated by something other than what the filter already predicts, so real
user speech during a reply doesn't get partially cancelled along with the echo, and
doesn't corrupt the filter's weights either. `barge_in_energy_threshold` /
`barge_in_cooldown_sec` (see above) remain as an on-top safety net either way — with
AEC on, they no longer need to be inflated to filter out raw echo, so they can stay
close to `energy_threshold`. Headphones still sidestep all of this entirely by
removing the acoustic path — the simplest fix when available.

## ASR hallucination on silence/noise

Even with AEC and the barge-in thresholds above tuned well, some residual echo/noise
will occasionally still clear `vad.energy_threshold` and `asr.min_speech_duration`
without being real speech. The failure mode this causes is not "nothing happens" —
Whisper (and derivatives like faster-whisper) is well documented to hallucinate
fluent, plausible-sounding filler text from near-silent or noisy audio instead of
transcribing nothing: stock phrases like "thanks for watching" in English, or
"Продолжение следует..." ("to be continued...") in Russian — confirmed directly
against this deployment by feeding it synthetic low-level noise. Whisper's own
`no_speech_prob` (only present in the `verbose_json` response format, one value per
segment) is a clean signal for this: the hallucinated noise transcription scored
above 0.89, genuine speech scored under 0.001.

Without filtering this, the hallucinated text reaches `llm` exactly as if the user
had said it, gets a real reply, that reply is spoken and partially picked back up by
the mic (no AEC is perfect), and the cycle can repeat — from the outside this looks
like "the assistant keeps talking and won't stop," generating generic
acknowledgement replies to a user who never actually said anything.

`OpenAICompatibleASRBackend`'s `no_speech_threshold` (default `0.6`, set on
`asr.asr_config` — see the config-keys table above) rejects a transcription whose
average `no_speech_prob` exceeds it, returning `""` instead of the hallucinated text.
`AsrMixin` still emits a `TextChunk` either way, but `LlmMixin` already drops any
chunk with blank `text` before it ever reaches the model (this check pre-dates the
voice assistant work), so a suppressed hallucination never starts a reply — no
further wiring was needed to close the loop. Set `no_speech_threshold: null` in
`asr_config` to disable the check entirely (e.g. against a server that doesn't
support `verbose_json`).

## Russian voice

`assets/configs/voice_assistant_pipeline_ru.yaml` is a Russian variant of the main
pipeline — same graph, same remote services, only the language-facing settings
differ: `asr_config.language: "ru"`, a Russian `system_prompt`, and
`tts_config.voice: "irina"`.

**Deployed and verified 2026-08-05** on `bleaf@192.168.1.153`: `ru_RU-irina-medium`
(a single-speaker Piper voice from the `rhasspy/piper-voices` Hugging Face repo) was
downloaded into `tts-voices/` and aliased as `"irina"` in
`tts-config/voice_to_speaker.yaml` (root-owned on that host — editing it needs
`sudo`), then the `tts` service was restarted. Round-trip tested for real:
`tts.voice="irina"` synthesized a valid 22050Hz mono WAV, `asr` (`language: "ru"`)
transcribed it back to essentially the same text, and `llm` answered a Russian
chat-completion prompt in Russian using this file's own `system_prompt`. `asr`/`llm`
needed no server-side changes (`faster-whisper-large-v3` is multilingual, Gemma
already understands Russian) — only the `tts` voice needed adding, since
`openedai-speech`'s default `voice_to_speaker.yaml` only ships the OpenAI-style
English aliases (alloy, echo, fable, onyx, nova, shimmer). The config file's header
comment has the same steps for adding a *different*/additional Russian Piper voice
later.

`asr_config.language` (`OpenAICompatibleASRBackend`, any `AsrNode`) is a general
addition, not Russian-specific: pinning it removes per-request language
auto-detection, which is one less thing that can misfire on a short utterance once an
assistant is only ever used in one language. Leave it unset (the default) to keep
auto-detection, e.g. for a genuinely multilingual setup.

## 3. Run it locally

Real microphone/speaker access needs live Python objects that plain YAML can't
express — the same reason `AudioReaderNode.from_config` already requires an injected
`device` (see `assets/configs/audio_vad_pipeline.yaml`). `AudioPlayerNode` follows the
identical pattern for output. `run_voice_assistant.py` (repo root) is the launch
script for this: it injects both devices by node type (works for either
`voice_assistant_pipeline.yaml` or the Russian `voice_assistant_pipeline_ru.yaml`,
or any other config with an `AudioReaderNode`/`AudioPlayerNode`), pre-flight checks
every remote service's `base_url` before starting anything, starts the control-plane
API in-process, and registers the pipeline with the dashboard — see its own
module docstring for the full behavior and every flag.

```bash
pip install -e ".[audio]"
python3 run_voice_assistant.py                # interactive config picker (TTY only)
python3 run_voice_assistant.py --config assets/configs/voice_assistant_pipeline_ru.yaml
```

Speak into the mic; once VAD detects silence after speech, you should hear a spoken
reply within a few seconds (network + inference latency on the remote box).

## 4. Watching it: metrics and the dashboard

`run_voice_assistant.py` does this by default — starts the
[control-plane API](../../README.md#-control-plane-api) (`neudc.service.api`)
**in the same process** (port `:8000` by default, `--api-port` to change it,
`--no-api` to skip it entirely) and registers the pipeline with the dashboard. Why
both have to happen there, for anyone building a different runner: the API must
share the pipeline's own process to see its Prometheus registry — a separately-run
`python -m neudc.entrypoints.api_server` would have none of this pipeline's metrics
in it — and `PipelineServiceManager` normally only knows about pipelines it started
itself via `/start` (a JSON-only contract this pipeline's live mic/speaker `device`
objects can't satisfy), so without an explicit `POST /pipelines/{name}/register` the
dashboard's pipeline list is just empty. `register_external_pipeline` only makes the
pipeline *visible* — the manager never starts, stops, or health-checks it — and
reports it as `"running"` for as long as it stays registered; the script calls the
mirror `POST /pipelines/{name}/unregister` on its own shutdown.

`GET /metrics` (raw Prometheus text, includes the `neudc_node_process_latency_seconds`
histogram) and `GET /ws/metrics` (a JSON snapshot pushed once a second) reflect
`reader`/`vad`/`asr`/`llm`/`tool`/`tts`/`player`'s real throughput/latency/errors as
they run. For the graphical dashboard, `frontend/` (`npm run dev`) shows the pipeline
at `/pipelines/<name>` (`<name>` defaults to the config's filename, e.g.
`voice-assistant` for `voice_assistant_pipeline.yaml`; `--pipeline-name` overrides it).

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

- **Resampling** if the deployed TTS voice can't emit 16 kHz natively.
- **A real VAD model.** Barge-in (see above) is built, but still rides on the same
  RMS-energy heuristic as before — good enough to demonstrate cancellation working
  end to end, not tuned against false positives from ambient noise.
