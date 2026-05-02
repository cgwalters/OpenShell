#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start a standalone openshell-gateway backed by the Podman compute driver for
# local manual testing.
#
# The gateway always starts in standard (supervised) mode. Sandbox isolation
# mode is a per-sandbox setting — clients select it at sandbox creation time.
#
# Defaults:
# - Plaintext HTTP on 127.0.0.1:18081
# - Dedicated sandbox namespace "podman-dev"
# - Persistent state under .cache/gateway-podman
# - Supervised mode (inner sandboxing active)
#
# Common overrides:
#   OPENSHELL_SERVER_PORT=19081 mise run gateway:podman
#   OPENSHELL_PODMAN_GATEWAY_NAME=my-podman-gateway mise run gateway:podman
#   OPENSHELL_SANDBOX_NAMESPACE=my-ns mise run gateway:podman
#   OPENSHELL_SANDBOX_IMAGE=ghcr.io/... mise run gateway:podman
#   OPENSHELL_SUPERVISOR_IMAGE=openshell/supervisor:dev mise run gateway:podman
#
# To create a sandbox with nested container support (inner Podman):
#   openshell sandbox create --image devenv-debian --mode nested
#
# With GCP Application Default Credentials for opencode:
#   OPENSHELL_PODMAN_ADC_PATH=~/.config/gcloud/application_default_credentials.json \
#   mise run gateway:podman
#
# Environment variable overrides:
#   OPENSHELL_SKIP_SUPERVISOR_CHECK=1  Skip supervisor image check (for nested-only gateway use)
#
# The ADC file is bind-mounted read-only at /run/gcloud/adc.json inside the
# container and GOOGLE_APPLICATION_CREDENTIALS is set automatically.
#
# After the gateway is running, point the CLI at it with either:
#   openshell --gateway podman-dev <command>
#   openshell gateway select podman-dev   # then plain `openshell <command>`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${OPENSHELL_SERVER_PORT:-18081}"
GATEWAY_NAME="${OPENSHELL_PODMAN_GATEWAY_NAME:-podman-dev}"
STATE_DIR="${OPENSHELL_PODMAN_GATEWAY_STATE_DIR:-${ROOT}/.cache/gateway-podman}"
SANDBOX_NAMESPACE="${OPENSHELL_SANDBOX_NAMESPACE:-podman-dev}"
SANDBOX_IMAGE="${OPENSHELL_SANDBOX_IMAGE:-ghcr.io/nvidia/openshell-community/sandboxes/base:latest}"
SANDBOX_IMAGE_PULL_POLICY="${OPENSHELL_SANDBOX_IMAGE_PULL_POLICY:-missing}"
SUPERVISOR_IMAGE="${OPENSHELL_SUPERVISOR_IMAGE:-openshell/supervisor:dev}"
LOG_LEVEL="${OPENSHELL_LOG_LEVEL:-info}"
GATEWAY_BIN="${ROOT}/target/debug/openshell-gateway"

# Optional: host path to a GCP Application Default Credentials JSON file.
# When set, the file is bind-mounted read-only into containers at
# /run/gcloud/adc.json and GOOGLE_APPLICATION_CREDENTIALS is set automatically.
# Example: OPENSHELL_PODMAN_ADC_PATH=~/.config/gcloud/application_default_credentials.json
ADC_PATH="${OPENSHELL_PODMAN_ADC_PATH:-}"

port_is_in_use() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${port}" >/dev/null 2>&1
    return $?
  fi
  (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1
}

register_gateway_metadata() {
  local name=$1
  local endpoint=$2
  local port=$3
  local config_home gateway_dir

  config_home="${XDG_CONFIG_HOME:-${HOME}/.config}"
  gateway_dir="${config_home}/openshell/gateways/${name}"

  mkdir -p "${gateway_dir}"
  cat >"${gateway_dir}/metadata.json" <<EOF
{
  "name": "${name}",
  "gateway_endpoint": "${endpoint}",
  "is_remote": false,
  "gateway_port": ${port},
  "auth_mode": "plaintext"
}
EOF
}

if [[ ! "${GATEWAY_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: OPENSHELL_PODMAN_GATEWAY_NAME must contain only letters, numbers, dots, underscores, or dashes" >&2
  exit 2
fi

if ! command -v podman >/dev/null 2>&1; then
  echo "ERROR: podman CLI is required" >&2
  exit 2
fi
if ! podman info >/dev/null 2>&1; then
  echo "ERROR: podman service is not reachable. Start it with:" >&2
  echo "  systemctl --user start podman.socket" >&2
  exit 2
fi

# The supervisor image must exist locally for supervised mode.
# Set OPENSHELL_SKIP_SUPERVISOR_CHECK=1 to skip this check (for nested-only gateway use).
if [[ "${OPENSHELL_SKIP_SUPERVISOR_CHECK:-}" == "1" ]]; then
  echo "WARNING: Skipping supervisor image check (OPENSHELL_SKIP_SUPERVISOR_CHECK=1)"
  echo "         Supervised sandboxes will fail if image '${SUPERVISOR_IMAGE}' is not present."
elif ! podman image exists "${SUPERVISOR_IMAGE}" 2>/dev/null; then
  echo "ERROR: supervisor image '${SUPERVISOR_IMAGE}' not found locally." >&2
  echo "Build it with: mise run build:docker:supervisor-load" >&2
  echo "Or set OPENSHELL_SKIP_SUPERVISOR_CHECK=1 to skip this check (nested-only use)." >&2
  exit 2
fi

if port_is_in_use "${PORT}"; then
  echo "ERROR: port ${PORT} is already in use; free it or set OPENSHELL_SERVER_PORT" >&2
  exit 2
fi

echo "Building openshell-gateway..."
cargo build -p openshell-server --bin openshell-gateway

if [[ ! -f "${GATEWAY_BIN}" ]]; then
  echo "ERROR: expected gateway binary at ${GATEWAY_BIN}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"

GATEWAY_ENDPOINT="http://127.0.0.1:${PORT}"
register_gateway_metadata "${GATEWAY_NAME}" "${GATEWAY_ENDPOINT}" "${PORT}"

# Generate a handshake secret for this session.
SSH_HANDSHAKE_SECRET="${OPENSHELL_SSH_HANDSHAKE_SECRET:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"

echo "Starting standalone Podman gateway (supervised mode)..."
echo "  gateway:          ${GATEWAY_NAME}"
echo "  endpoint:         ${GATEWAY_ENDPOINT}"
echo "  namespace:        ${SANDBOX_NAMESPACE}"
echo "  sandbox image:    ${SANDBOX_IMAGE}"
echo "  supervisor image: ${SUPERVISOR_IMAGE}"
if [[ -n "${ADC_PATH}" ]]; then
  echo "  adc host path:    ${ADC_PATH}"
fi
echo "  state dir:        ${STATE_DIR}"
echo
echo "Point the CLI at this gateway with one of:"
echo "  openshell --gateway ${GATEWAY_NAME} status"
echo "  openshell gateway select ${GATEWAY_NAME}"
echo
echo "To create a sandbox with nested container support:"
echo "  openshell sandbox create --image devenv-debian --mode nested"
echo

OPENSHELL_SSH_HANDSHAKE_SECRET="${SSH_HANDSHAKE_SECRET}" \
OPENSHELL_SUPERVISOR_IMAGE="${SUPERVISOR_IMAGE}" \
OPENSHELL_PODMAN_ADC_PATH="${ADC_PATH}" \
exec "${GATEWAY_BIN}" \
  --port "${PORT}" \
  --log-level "${LOG_LEVEL}" \
  --drivers podman \
  --disable-tls \
  --db-url "sqlite:${STATE_DIR}/gateway.db?mode=rwc" \
  --sandbox-namespace "${SANDBOX_NAMESPACE}" \
  --sandbox-image "${SANDBOX_IMAGE}" \
  --sandbox-image-pull-policy "${SANDBOX_IMAGE_PULL_POLICY}" \
  --ssh-gateway-port "${PORT}"
