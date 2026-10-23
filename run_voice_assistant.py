"""Launch script for a voice-assistant-shaped pipeline.

Live devices, a config picker, a remote-service pre-flight check, and dashboard
wiring, all in one entrypoint. Mirrors neudc.entrypoints.main.main(), adapted the way
docs/voice_assistant/README.md
documents: AudioReaderNode/AudioPlayerNode need live device objects that plain YAML
can't express, so this script builds them and injects them before NodeFactory.create.
Device injection is driven by node *type* (AudioReaderNode / AudioPlayerNode), not a
fixed node id, so this works for any config under assets/configs/ that happens to
have one — not just assets/configs/voice_assistant_pipeline.yaml (the Russian variant,
assets/configs/voice_assistant_pipeline_ru.yaml, works the same way).

Before starting any node, it also probes every *_config.base_url in the chosen config
(asr_config/llm_config/tts_config — whichever the config actually has) with a plain
GET {base_url}/models, and refuses to start if any of them isn't reachable: the
alternative is nodes starting fine and then failing mysteriously mid-conversation on
the first HTTP call, which is a much more confusing failure to diagnose live. Pass
--skip-health-check to start anyway (e.g. against a service that doesn't expose
/models).

Also starts the control-plane API (neudc.service.api) in this same process, on
--api-port (default 8000). It has to run *in-process* rather than as the separate
`python -m neudc.entrypoints.api_server` server: that server's /pipelines endpoints
manage pipelines it starts itself from JSON config (PipelineServiceManager), which
can't express live mic/speaker device objects either — so it would have nothing to
show for this pipeline. Running it here means /metrics and /ws/metrics read from the
same process-local Prometheus registry these nodes are actually recording into.

It also *registers* this pipeline over HTTP (POST /pipelines/{name}/register) using a
device-free copy of the node config, purely so the dashboard (frontend/, `npm run
dev`, or `docker compose up frontend` with
API_BASE_URL=http://host.docker.internal:8000) can see it — the manager never
starts/stops it, this script still owns that.

Usage:
    python3 run_voice_assistant.py                      # interactive config picker (TTY only)
    python3 run_voice_assistant.py --config PATH         # skip the picker, use this config
    python3 run_voice_assistant.py --list                # list available configs and exit
    python3 run_voice_assistant.py --smoke                # starts, waits a couple seconds, stops itself
    python3 run_voice_assistant.py --skip-health-check    # start even if a remote service check fails
    python3 run_voice_assistant.py --api-port 8000        # control-plane API port (default 8000)
    python3 run_voice_assistant.py --pipeline-name NAME   # name shown on the dashboard (default: config filename)
    python3 run_voice_assistant.py --speaker-sample-rate 22050  # AudioPlayerNode output rate (default 22050)
    python3 run_voice_assistant.py --no-api               # skip starting the API entirely
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from neudc.core.communication.messaging.routing_factory import RoutingFactory
from neudc.core.node.broadcast.audio_player import SoundDeviceAudioOutputDevice
from neudc.core.node.node_factory import NodeFactory
from neudc.core.node.readers.audio_reader import SoundDeviceAudioDevice
from neudc.core.utils.config_loader import load_config
from neudc.core.utils.config_schema import validate_pipeline_config

CONFIGS_DIR = Path("assets/configs")
DEFAULT_CONFIG_NAME = "voice_assistant_pipeline.yaml"
#: Node config keys carrying a remote OpenAI-compatible service's base_url, checked
#: before startup — matches AsrNode/LlmNode/TtsNode's own `from_config` conventions.
REMOTE_SERVICE_CONFIG_KEYS = ("asr_config", "llm_config", "tts_config")


def _list_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def _pick_config(explicit: str | None) -> Path:
    """Resolve which config to run: --config if given, else a picker (TTY) or the default."""
    if explicit:
        return Path(explicit)

    configs = _list_configs()
    if not configs:
        msg = f"No YAML configs found under {CONFIGS_DIR}/"
        raise SystemExit(msg)
    default = next((c for c in configs if c.name == DEFAULT_CONFIG_NAME), configs[0])

    if not sys.stdin.isatty():
        print(f"Non-interactive session, defaulting to {default} (pass --config to choose explicitly)")
        return default

    print("Available pipeline configs:")
    for i, c in enumerate(configs, 1):
        marker = "  (default)" if c == default else ""
        print(f"  {i}. {c.name}{marker}")
    choice = input(f"Pick a config [1-{len(configs)}, Enter for default]: ").strip()
    if not choice:
        return default
    if not choice.isdigit() or not (1 <= int(choice) <= len(configs)):
        print(f"Invalid choice, using default: {default.name}")
        return default
    return configs[int(choice) - 1]


def _check_remote_services(config: dict[str, Any], *, timeout: float = 5.0) -> list[str]:
    """Probe every node's `*_config.base_url` with `GET {base_url}/models`.

    Returns a list of human-readable failure messages (empty if every service
    responded). Only checks node types that actually carry a `base_url` — a config
    with none (e.g. a plain CV pipeline) is a no-op, same as `_inject_audio_devices`.
    """
    failures: list[str] = []
    for node_config in config["nodes"]:
        for key in REMOTE_SERVICE_CONFIG_KEYS:
            service_config = node_config.get(key)
            if not isinstance(service_config, dict):
                continue
            base_url = service_config.get("base_url")
            if not base_url:
                continue
            label = f"{node_config['id']} ({key}: {base_url})"
            try:
                response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
                response.raise_for_status()
            except requests.RequestException as e:
                failures.append(f"{label}: {e}")
            else:
                print(f"Remote service OK: {label}")
    return failures


def _inject_audio_devices(config: dict[str, Any], *, speaker_sample_rate: int) -> None:
    """Inject a live SoundDevice object into any AudioReaderNode/AudioPlayerNode, by type.

    A no-op for configs with neither node type (e.g. a plain CV pipeline) — nothing
    here assumes the graph is audio-shaped.
    """
    for node_config in config["nodes"]:
        node_type = node_config.get("type")
        if node_type == "AudioReaderNode":
            node_config["device"] = SoundDeviceAudioDevice(
                sample_rate=node_config.get("sample_rate", 16000),
                channels=node_config.get("channels", 1),
            )
            print(f"Injected microphone device for node '{node_config['id']}'")
        elif node_type == "AudioPlayerNode":
            node_config["device"] = SoundDeviceAudioOutputDevice(sample_rate=speaker_sample_rate, channels=1)
            print(f"Injected speaker device for node '{node_config['id']}' at {speaker_sample_rate} Hz")


def _start_api_server(port: int) -> None:
    """Start the control-plane FastAPI app (uvicorn) in a background daemon thread."""
    import uvicorn

    from neudc.service.api import create_app

    app = create_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")  # noqa: S104 - local dev tool
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    print(f"Control-plane API on http://127.0.0.1:{port} (docs: /docs, metrics: /metrics, live: /ws/metrics)")


def _register_with_dashboard(base_url: str, name: str, nodes_config: list[dict[str, Any]]) -> None:
    """POST /pipelines/{name}/register, retrying briefly while uvicorn finishes binding.

    The *first* call to validate_pipeline_config in this process lazily imports
    NodeFactory's node modules, some of which are numba-jitted (compiled on first
    import, not cached) — that warmup can take several seconds server-side. The
    per-request timeout has to comfortably clear that, or a slow-but-successful first
    attempt looks like a failure client-side and the retry lands on a 409 (already
    registered) instead of the 201 it should have seen.
    """
    deadline = time.time() + 30.0
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = requests.post(f"{base_url}/pipelines/{name}/register", json={"nodes": nodes_config}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            last_error = e
            time.sleep(0.2)
        else:
            print(f"Registered with the dashboard as '{name}' (frontend/: npm run dev, then /pipelines/{name})")
            return
    print(f"Could not register with the dashboard (continuing without it): {last_error}")


def _unregister_from_dashboard(base_url: str, name: str) -> None:
    try:
        requests.post(f"{base_url}/pipelines/{name}/unregister", timeout=2)
    except requests.RequestException as e:
        print(f"Could not unregister from the dashboard (harmless, process is exiting): {e}")


def main() -> None:
    """Parse CLI args, run the pre-flight checks, then build and run the pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to a pipeline YAML (skips the interactive picker)")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Start even if a remote service (asr/llm/tts base_url) isn't reachable",
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--pipeline-name", help="Name shown on the dashboard (default: config filename)")
    parser.add_argument("--speaker-sample-rate", type=int, default=22050)
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    if args.list:
        for c in _list_configs():
            print(c)
        return

    config_path = _pick_config(args.config)
    pipeline_name = args.pipeline_name or config_path.stem.replace("_", "-")

    config = load_config(config_path)
    validate_pipeline_config(config)
    # Snapshot the node list *before* device injection below — it must stay JSON-safe
    # (no live SoundDevice objects) to hand to register_external_pipeline/the dashboard.
    display_nodes_config = load_config(config_path)["nodes"]

    if not args.skip_health_check:
        failures = _check_remote_services(config)
        if failures:
            print("Remote service check FAILED:")
            for failure in failures:
                print(f"  - {failure}")
            msg = "Refusing to start with an unreachable remote service (pass --skip-health-check to override)."
            raise SystemExit(msg)

    _inject_audio_devices(config, speaker_sample_rate=args.speaker_sample_rate)

    router = RoutingFactory(config)
    mailbox_map = router.create_mailboxes()
    nodes = [
        NodeFactory.create(node_config, mailbox=mailbox_map[node_config["id"]][0]) for node_config in config["nodes"]
    ]

    api_base_url = f"http://127.0.0.1:{args.api_port}"
    if not args.no_api:
        _start_api_server(args.api_port)
        _register_with_dashboard(api_base_url, pipeline_name, display_nodes_config)

    def _unregister() -> None:
        if not args.no_api:
            _unregister_from_dashboard(api_base_url, pipeline_name)

    print(f"Running {config_path} — starting {len(nodes)} nodes: {[n.id for n in nodes]}", flush=True)
    for node in nodes:
        node.start()
    print("All nodes started.", flush=True)

    if args.smoke:
        time.sleep(3)
        print("Smoke test: nodes ran for 3s without crashing, stopping now.", flush=True)
        for node in nodes:
            node.stop()
        _unregister()
        failed = [n.id for n in nodes if n.failed()]
        if failed:
            print(f"FAILED nodes: {failed}", flush=True)
            raise SystemExit(1)
        print("Smoke test OK.", flush=True)
        return

    print("Pipeline running. Press Ctrl+C to stop.", flush=True)
    stop_event = threading.Event()

    def _handle_sigint(signum: int, frame: object) -> None:
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    stop_event.wait()

    print("Stopping...", flush=True)
    for node in nodes:
        node.stop()
    _unregister()
    print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
