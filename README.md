# fal-native-hosting

Deploy four [fal](https://fal.ai) native microservices that split inference across **preprocessing**, **removebg**, **colmap**, and **postprocessing** apps, and route model calls from Python via `native_fal_api.py` (msgpack over HTTP to each service’s `/infer/...` endpoints).

## Prerequisites

- [fal CLI](https://fal.ai/docs) installed and authenticated (defaults to `FAL_BIN=/home/spyne-4090/miniconda3/envs/fal/bin/fal` in the shell scripts; override with `FAL_BIN` if yours differs).
- A local or mounted **model weights** tree; scripts and apps expect `MODELS_SOURCE` in `.env` (Triton-style layout used by the fal apps).

## Configuration

Create a `.env` in this directory (it is gitignored). Typical variables:

| Variable | Purpose |
| --- | --- |
| `FAL_KEY` | API key for authenticated requests (required for `native_fal_api.py` and warm-up curls). |
| `MODELS_SOURCE` | Path to the model repository / weights directory on the host used during deploy. |
| `HEAVY_MODELS` | Comma-separated model ids routed to the “heavy” machine type where supported. |
| `MAIN_MACHINE_TYPE` | Default GPU/machine type for most apps (e.g. `GPU-RTX4090`). |
| `HEAVY_MACHINE_TYPE` | Machine type for heavy models. |
| `KEEP_ALIVE` | Runner keep-alive seconds (as used by the fal app definitions). |
| `FAL_TEAM` | Optional; passed to `fal apps delete` in `teardown_apps.sh` if the default team is wrong. |

After a successful deploy, `deploy.sh` writes **`endpoints.env`** with `FAL_ENDPOINT_PREPROCESS`, `FAL_ENDPOINT_REMOVEBG`, `FAL_ENDPOINT_COLMAP`, and `FAL_ENDPOINT_POSTPROCESS`. Load that file alongside `.env` when running clients that call the hosted services.

## Scripts

| Script | What it does |
| --- | --- |
| `./deploy.sh` | Deploys all four apps with the fal CLI and regenerates `endpoints.env`. |
| `./spin_up_native.sh` | Hits each service’s `/health` sequentially until HTTP 200 (cold starts can take many minutes). Requires existing `endpoints.env`. |
| `./run_all_native.sh` | Runs `deploy.sh` then `spin_up_native.sh`. Use `SKIP_DEPLOY=1` to warm up only; `FULL_REBUILD=1` to run `rebuild_all.sh` first. |
| `./rebuild_all.sh` | `teardown_apps.sh` then `deploy.sh` with `FAL_DEPLOY_NO_CACHE=1` (full image rebuild; long run). |
| `./teardown_apps.sh` | Deletes the four fal apps (and legacy `serve`) for a clean redeploy. |

Optional: `FAL_DEPLOY_NO_CACHE=1` on `deploy.sh` forces `--no-cache` without a full teardown.

## Python client

Import and use `fal_infer` from `native_fal_api.py` like the monolithic `fal_api` helper: set `FAL_KEY` and the `FAL_ENDPOINT_*` variables (from `endpoints.env`), then call `fal_infer(...)` with numpy inputs; routing from model name to microservice is defined in `MODEL_ROUTING` inside that module.

## Repository layout

- `src/apps/*_impl.py` — FastAPI / inference implementation per service.
- `src/apps/*.py` — Thin fal entrypoints that delegate to `*_impl` (keeps isolate serialization small).
- `src/utils.py` — Shared helpers.
- `inspect_storage.py` — Optional storage inspection utility.
