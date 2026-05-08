#!/usr/bin/env bash
set -euo pipefail

# Deploy all native fal microservices, then warm them sequentially (same as deploy.sh + spin_up_native.sh).
#
# Optional env:
#   SKIP_DEPLOY=1     — only run spin_up_native.sh (endpoints.env must already exist)
#   FULL_REBUILD=1    — run rebuild_all.sh (teardown + FAL_DEPLOY_NO_CACHE deploy) then warm-up
#   FAL_DEPLOY_NO_CACHE=1 — full rebuild when deploy runs (no teardown); see deploy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

echo "========================================="
echo "Native fal: deploy + warm-up"
echo "========================================="

if [[ "${FULL_REBUILD:-0}" == "1" ]]; then
    echo "FULL_REBUILD=1 — teardown + deploy with --no-cache..."
    ./rebuild_all.sh
elif [[ "${SKIP_DEPLOY:-0}" == "1" ]]; then
    echo "SKIP_DEPLOY=1 — skipping deploy.sh"
else
    ./deploy.sh
fi

./spin_up_native.sh

echo "========================================="
echo "Native fal pipeline finished."
echo "========================================="
