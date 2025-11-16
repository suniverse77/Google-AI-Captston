#!/bin/bash
set -euo pipefail

# WinCLIP baseline 전 카테고리 일괄 실행 스크립트.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/mvtec_winclip.yaml}"
DEVICE="${DEVICE:-}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "[INFO] Using config: $CONFIG_PATH"

readarray -t CATEGORIES < <(python - <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp) or {}

categories = cfg.get("data", {}).get("categories") or ["bottle"]
for category in categories:
    print(category)
PY
"$CONFIG_PATH")

for category in "${CATEGORIES[@]}"; do
  echo "[INFO] Running category: $category"
  CATEGORY_ENV="WINCLIP_CATEGORY=$category"
  if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG="--device $DEVICE"
  else
    DEVICE_ARG=""
  fi
  WINCLIP_CATEGORY="$category" python "$ROOT_DIR/src/main.py" --config "$CONFIG_PATH" $DEVICE_ARG
done

