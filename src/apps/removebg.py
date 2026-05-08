"""fal entrypoint only; app in removebg_impl. See preprocessing.py for COPY/context notes."""
import textwrap
from pathlib import Path

import inspect

import fal

_REPO_ROOT = Path(__file__).resolve().parents[2]

_IMPL = "src.apps.removebg_impl"


def _ensure_syspath(caller_globals: dict, tag: str) -> None:
    import os
    import sys

    def add_front(p: str) -> None:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # Image uses WORKDIR /app, COPY src /app/src, PYTHONPATH=/app
    if os.path.isfile("/app/src/utils.py") or os.path.isdir("/app/src"):
        add_front("/app")
    elif os.environ.get("PYTHONPATH", "").startswith("/app") or ":/app" in os.environ.get(
        "PYTHONPATH", ""
    ):
        add_front("/app")
    env = os.environ.get("FAL_NATIVE_PROJECT_ROOT", "").strip()
    if env and os.path.isdir(os.path.join(env, "src")):
        add_front(os.path.abspath(env))

    ef = caller_globals.get("__file__")
    if ef:
        d = os.path.dirname(os.path.abspath(ef))
        for _ in range(24):
            if os.path.isfile(os.path.join(d, "src", "utils.py")):
                add_front(d)
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    print(
        f"[fal-native/{tag}] cwd={os.getcwd()!r} PYTHONPATH={os.environ.get('PYTHONPATH')!r} "
        f"has_app_src_utils={os.path.isfile('/app/src/utils.py')!r} "
        f"sys.path0={sys.path[0]!r}",
        flush=True,
    )


@fal.function(
    machine_type="GPU-RTX5090",
    image=fal.ContainerImage.from_dockerfile_str(
        textwrap.dedent(
            """
            FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

            ENV DEBIAN_FRONTEND=noninteractive
            RUN apt-get update && apt-get install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0

            RUN pip3 install --upgrade pip
            RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
            RUN pip3 install fastapi uvicorn msgpack msgpack-numpy numpy onnxruntime-gpu

            WORKDIR /app
            COPY src /app/src
            ENV PYTHONPATH=/app
            """
        ).strip(),
        context_dir=_REPO_ROOT,
    ),
    exposed_port=8000,
    keep_alive=300,
    startup_timeout=1800,
)
def removebg():
    import importlib
    import sys
    import uvicorn

    print("[fal-native/removebg] entry", flush=True)
    _ensure_syspath(inspect.currentframe().f_globals, "removebg")

    svc = importlib.import_module(_IMPL)
    print("[fal-native/removebg] uvicorn on 0.0.0.0:8000", flush=True)
    uvicorn.run(svc.app, host="0.0.0.0", port=8000)
