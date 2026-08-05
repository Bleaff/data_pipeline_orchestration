# Standalone control-plane API (neudc.entrypoints.api_server) — see docker-compose.yml.
#
# This is for pipelines PipelineServiceManager can start itself via POST
# /pipelines/{name}/start (a JSON-only config, no live device objects). It is NOT
# how the voice assistant pipeline runs: that needs a real microphone/speaker, which
# a container can't reach here — Docker Desktop on macOS runs containers in a Linux
# VM with no CoreAudio access. Run the voice assistant on the host instead (see
# docs/voice_assistant/README.md) and point the `frontend` service at its embedded
# API (API_BASE_URL=http://host.docker.internal:8000), not at this container.
#
# Image size note: neudc's hard dependencies (torch, torchvision, ultralytics,
# opencv-python-headless, numba, ...) are not split into optional extras, so even
# this API-only image pulls all of them regardless — expect a slow first build.
# torch is at least kept CPU-only (see below): this container has no GPU access
# either way (Docker Desktop on macOS has none at all; this image isn't built with
# --gpus in mind on Linux either), so pulling PyPI's default CUDA-bundled linux
# wheel would cost several extra GB of nvidia-* packages for nothing.

FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY neudc ./neudc

# Install the CPU-only torch/torchvision build *first*, from PyTorch's own CPU wheel
# index — PyPI's default linux wheel for `torch` bundles the full CUDA runtime
# (nvidia-cublas, nvidia-cudnn, ...) as dependencies, several GB this container will
# never use. Once a version satisfying pyproject.toml's `torch = "^2.2.1"` is already
# installed, the next step's resolver sees the constraint is met and leaves it alone.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# `api` extra: uvicorn + websockets, the only deps the control-plane server needs
# beyond neudc's (already substantial) hard requirements. No audio/dev/trt/onnx —
# this container never touches a microphone or a GPU.
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["python3", "-m", "neudc.entrypoints.api_server", "--host", "0.0.0.0", "--port", "8000"]
