#!/usr/bin/env bash
set -euo pipefail

# Warm one runner per native app (sequential — avoids 4 GPU cold-starts at once and
# interleaved curl output). First GPU cold start can take many minutes after deploy.
#
# Optional env:
#   SPINUP_MAX_TIME    seconds per curl (default 600)
#   SPINUP_CONNECT_TIMEOUT  (default 60)
#   SPINUP_ATTEMPTS    retries per app (default 20)
#   SPINUP_SLEEP       sleep between retries (default 25)
#   SPINUP_DEBUG=1     verbose curl to stderr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SPINUP_MAX_TIME="${SPINUP_MAX_TIME:-600}"
SPINUP_CONNECT_TIMEOUT="${SPINUP_CONNECT_TIMEOUT:-60}"
SPINUP_ATTEMPTS="${SPINUP_ATTEMPTS:-20}"
SPINUP_SLEEP="${SPINUP_SLEEP:-25}"

if [[ -f endpoints.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source endpoints.env
    set +a
else
    echo "Error: endpoints.env not found. Run deploy.sh first." >&2
    exit 1
fi

if [[ -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

if [[ -z "${FAL_KEY:-}" ]]; then
    echo "Warning: FAL_KEY is not set; expect 401 from private apps." >&2
fi

curl_base=(
    -sS
    --connect-timeout "${SPINUP_CONNECT_TIMEOUT}"
    --max-time "${SPINUP_MAX_TIME}"
    -o /dev/null
    -w "%{http_code}"
    -H "Authorization: Key ${FAL_KEY:-}"
)

warm_one() {
    local label="$1"
    local var_name="$2"
    local base="${!var_name:-}"
    if [[ -z "$base" ]]; then
        echo "Error: ${var_name} is not set (check endpoints.env)." >&2
        return 1
    fi
    local url="${base%/}/health"

    echo ""
    echo "==> ${label}  ${url}"
    echo "    (max ${SPINUP_MAX_TIME}s per attempt, ${SPINUP_ATTEMPTS} attempts, ${SPINUP_SLEEP}s between)"

    local i http_code curl_ec
    for i in $(seq 1 "${SPINUP_ATTEMPTS}"); do
        set +e
        http_code="$(curl "${curl_base[@]}" "$url" 2>/dev/null)"
        curl_ec=$?
        set -e

        if [[ "${curl_ec}" -eq 0 && "${http_code}" == "200" ]]; then
            echo "    OK (HTTP 200) after attempt ${i}"
            return 0
        fi

        if [[ "${curl_ec}" -eq 0 && "${http_code}" == "401" ]]; then
            echo "    FAIL: HTTP 401 — check FAL_KEY and app auth." >&2
            return 1
        fi

        if [[ "${curl_ec}" -eq 0 ]]; then
            echo "    attempt ${i}: HTTP ${http_code} (waiting on runner / routing...)"
        else
            echo "    attempt ${i}: curl exit ${curl_ec} (timeout or connection error — GPU provision often still in progress)"
        fi
        sleep "${SPINUP_SLEEP}"
    done

    echo "    FAIL: no HTTP 200 from /health for ${label}" >&2
    return 1
}

echo "========================================="
echo "Native fal warm-up (sequential)"
echo "Tip: finish deploy.sh first; A100/RTX5090 slots can take 10+ min on first request."
echo "========================================="

FAIL=0
warm_one "preprocessing" "FAL_ENDPOINT_PREPROCESS" || FAIL=1
warm_one "removebg" "FAL_ENDPOINT_REMOVEBG" || FAIL=1
warm_one "colmap" "FAL_ENDPOINT_COLMAP" || FAIL=1
warm_one "postprocessing" "FAL_ENDPOINT_POSTPROCESS" || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
    echo ""
    echo "One or more warm-ups did not get HTTP 200." >&2
    echo "Check fal dashboard logs for each app; if deploy is still running, wait and retry." >&2
    echo "If runners never appear, try a smaller machine_type temporarily (e.g. GPU-A100 for all)." >&2
    exit 1
fi

echo ""
echo "All four apps returned HTTP 200 on /health."
