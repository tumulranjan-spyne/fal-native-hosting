#!/usr/bin/env bash
set -euo pipefail

# Remove native microservice apps from fal so the next deploy is a fresh alias/runtime.
# Optional: set FAL_TEAM in .env if your CLI default team is not the one that owns these apps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

FAL_BIN="${FAL_BIN:-/home/spyne-4090/miniconda3/envs/fal/bin/fal}"

if ! command -v "$FAL_BIN" &> /dev/null; then
    echo "Error: fal CLI not found." >&2
    exit 1
fi

TEAM_ARGS=()
if [[ -n "${FAL_TEAM:-}" ]]; then
    TEAM_ARGS=(--team "${FAL_TEAM}")
fi

delete_one() {
    local name="$1"
    echo "Deleting fal app: ${name} ..."
    if out=$("${FAL_BIN}" apps delete "${TEAM_ARGS[@]}" "${name}" 2>&1); then
        echo "${out}"
    else
        echo "${out}" >&2
        echo "  (continuing — app may not exist or name/team mismatch)" >&2
    fi
}

echo "========================================="
echo "Tearing down native fal apps"
echo "========================================="

# Current four services
delete_one "preprocessing"
delete_one "removebg"
delete_one "colmap"
delete_one "postprocessing"

# Legacy monolithic app name from earlier iterations
delete_one "serve"

echo "========================================="
echo "Teardown pass complete."
echo "Run ./deploy.sh (or FAL_DEPLOY_NO_CACHE=1 ./rebuild_all.sh workflow) to redeploy."
echo "========================================="
