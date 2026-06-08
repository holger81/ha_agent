#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HA_CONFIG="${HA_CONFIG:-}"

if [[ -z "${HA_CONFIG}" ]]; then
  for candidate in \
    "${HOME}/homeassistant/config" \
    "${HOME}/.homeassistant" \
    "/config"; do
    if [[ -d "${candidate}" ]]; then
      HA_CONFIG="${candidate}"
      break
    fi
  done
fi

if [[ -z "${HA_CONFIG}" || ! -d "${HA_CONFIG}" ]]; then
  echo "Set HA_CONFIG to your Home Assistant config directory." >&2
  exit 1
fi

TARGET="${HA_CONFIG}/custom_components/ha_agent"
mkdir -p "${HA_CONFIG}/custom_components"
rsync -av --delete "${ROOT_DIR}/custom_components/ha_agent/" "${TARGET}/"

echo "Deployed ha_agent to ${TARGET}"
echo "Restart Home Assistant to load the integration."
