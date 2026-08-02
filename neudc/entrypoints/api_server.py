"""Entry point for the control-plane API server: `python -m neudc.entrypoints.api_server`."""

from __future__ import annotations

import argparse

import uvicorn

from neudc.service.api import create_app


def main() -> None:
    """Parse CLI args and serve the control-plane API."""
    parser = argparse.ArgumentParser(description="Run the neudc control-plane API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
