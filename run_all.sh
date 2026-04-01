#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_lora_pipeline.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

PIPELINE_TYPE="$(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config
config = load_yaml_config(sys.argv[1])
print(config["pipeline_type"])
PY
)"

case "$PIPELINE_TYPE" in
  inference_only)
    exec ./scripts/run_inference_pipeline.sh "$CONFIG_PATH"
    ;;
  lora_train_eval)
    exec ./scripts/run_lora_pipeline.sh "$CONFIG_PATH"
    ;;
  *)
    echo "Unsupported pipeline_type in ${CONFIG_PATH}: ${PIPELINE_TYPE}" >&2
    exit 1
    ;;
esac
