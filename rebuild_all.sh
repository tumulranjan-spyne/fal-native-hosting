#!/usr/bin/env bash
set -euo pipefail

# Delete all native microservice apps, then deploy from scratch with --no-cache
# (forces a full container rebuild; expect a long run).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/teardown_apps.sh"
export FAL_DEPLOY_NO_CACHE=1
"${SCRIPT_DIR}/deploy.sh"
