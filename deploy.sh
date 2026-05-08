#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

FAL_BIN="${FAL_BIN:-/home/spyne-4090/miniconda3/envs/fal/bin/fal}"

if ! command -v "$FAL_BIN" &> /dev/null; then
    echo "Error: fal CLI not found." >&2
    exit 1
fi

echo "========================================="
echo "Deploying Native Fal Microservices"
echo "========================================="

if [[ "${FAL_DEPLOY_NO_CACHE:-0}" == "1" ]]; then
    echo "FAL_DEPLOY_NO_CACHE=1 — passing --no-cache (full environment rebuild)."
    DEPLOY_EXTRA_FLAGS=(--no-cache)
else
    DEPLOY_EXTRA_FLAGS=()
fi

extract_sync_url() {
    local log="$1"
    local app_name="$2"
    
    # Prefer sync host (fal.run); fal also prints queue.fal.run -- use that only as fallback.
    local url
    url=$(echo "$log" | grep -oE 'https://fal\.run/[^[:space:]]*' | head -1 || true)
    if [[ -z "$url" ]]; then
        url=$(echo "$log" | grep -oE 'https://[a-zA-Z0-9][-a-zA-Z0-9.]*\.fal\.run[^[:space:]]*' | head -1 || true)
    fi
    
    if [[ -z "$url" ]]; then
        echo "Error: Could not extract URL for ${app_name}" >&2
        return 1
    fi
    echo "$url"
}

deploy_app() {
    local file_path="$1"
    local app_name="$2"
    local env_var_name="$3"
    
    echo "Deploying ${app_name}..."
    local log_file="deploy-${app_name}.log"
    
    # Deploy and capture output
    if ! "$FAL_BIN" deploy "${file_path}::${app_name}" --app-name "${app_name}" \
        --reset-scale --yes "${DEPLOY_EXTRA_FLAGS[@]}" 2>&1 | tee "$log_file"; then
        echo "Error: Deployment failed for ${app_name}." >&2
        return 1
    fi
    
    local url
    url=$(extract_sync_url "$(cat "$log_file")" "$app_name")
    
    if [[ -n "$url" ]]; then
        echo "✅ ${app_name} deployed successfully: ${url}"
        echo "${env_var_name}=${url}" >> endpoints.env
    else
        return 1
    fi
}

# Clear old endpoints
> endpoints.env

deploy_app "src/apps/preprocessing.py" "preprocessing" "FAL_ENDPOINT_PREPROCESS"
deploy_app "src/apps/removebg.py" "removebg" "FAL_ENDPOINT_REMOVEBG"
deploy_app "src/apps/colmap.py" "colmap" "FAL_ENDPOINT_COLMAP"
deploy_app "src/apps/postprocessing.py" "postprocessing" "FAL_ENDPOINT_POSTPROCESS"

echo "========================================="
echo "Deployment Complete!"
echo "Endpoints saved to endpoints.env:"
cat endpoints.env
echo "========================================="
