#!/usr/bin/env bash
# Start a new user-controlled session with network access for Modal and downloads.
# This does not change the current session or override managed restrictions.
set -euo pipefail

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

exec codex \
  --cd "$experiment_root" \
  --sandbox workspace-write \
  -c 'sandbox_workspace_write.network_access=true' \
  "$@"
