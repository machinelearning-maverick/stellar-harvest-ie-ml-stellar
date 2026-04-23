#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# stellar-harvest-ie-ml-stellar/publish-module.sh
# ------------------------------------------------------------
# Delegates to the central publish script in stellar-harvest-ie-deployment.
# Usage:
#   ./publish-module.sh                  # build & upload all dist/*
#   ./publish-module.sh --version 0.2.1  # upload only dist/*0.2.1*
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CENTRAL_SCRIPT="$(dirname "$SCRIPT_DIR")/stellar-harvest-ie-deployment/deployment/scripts/publish-module.sh"

if [[ ! -f "$CENTRAL_SCRIPT" ]]; then
  echo "Central publish script not found at: $CENTRAL_SCRIPT"
  echo "Make sure stellar-harvest-ie-deployment is cloned as a sibling of this module."
  exit 1
fi

exec bash "$CENTRAL_SCRIPT" "$@" "$SCRIPT_DIR"