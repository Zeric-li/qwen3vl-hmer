#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_inference_pipeline.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

readarray -t CONFIG_VALUES < <(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config

config = load_yaml_config(sys.argv[1])
if config.get("pipeline_type") != "inference_only":
    raise SystemExit(f"Expected pipeline_type=inference_only, got {config.get('pipeline_type')}")

print(config["model_id"])
print(config["output_dir"])
print(config.get("eval_dataset_id", config.get("train_dataset_id", "Neeze/CROHME-full")))
print("" if config.get("max_eval_samples") is None else config["max_eval_samples"])
print(config["cdm_toolkit_path"])
print(config.get("cdm_pools", 1))
for split in config.get("eval_splits", []):
    print(f"SPLIT::{split}")
PY
)

MODEL_ID="${CONFIG_VALUES[0]}"
OUTPUT_DIR="${CONFIG_VALUES[1]}"
DATASET_ID="${CONFIG_VALUES[2]}"
MAX_EVAL_SAMPLES="${CONFIG_VALUES[3]}"
CDM_TOOLKIT_PATH="${CONFIG_VALUES[4]}"
CDM_POOLS="${CONFIG_VALUES[5]}"

EVAL_SPLITS=()
EVAL_DIRS=()
for value in "${CONFIG_VALUES[@]:6}"; do
  EVAL_SPLITS+=("${value#SPLIT::}")
done

if [[ "${#EVAL_SPLITS[@]}" -eq 0 ]]; then
  echo "Config must define at least one eval_splits entry: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "$CDM_TOOLKIT_PATH" ]]; then
  echo "UniMERNet CDM toolkit not found: ${CDM_TOOLKIT_PATH}" >&2
  echo "Clone UniMERNet to ./external/UniMERNet and install external/UniMERNet/cdm/requirements.txt." >&2
  exit 1
fi

for required_cmd in node convert pdflatex; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    echo "Required command for UniMERNet CDM not found: ${required_cmd}" >&2
    echo "Install Node.js, ImageMagick, and a LaTeX distribution before running evaluation." >&2
    exit 1
  fi
done

for EVAL_SPLIT in "${EVAL_SPLITS[@]}"; do
  EVAL_DIR="${OUTPUT_DIR}/base_model_eval_${EVAL_SPLIT}"

  INFER_CMD=(
    python -m scripts.run_inference
    --checkpoint "$MODEL_ID"
    --output-dir "$EVAL_DIR"
    --batch-size "${BASE_EVAL_BATCH_SIZE:-1}"
    --split "$EVAL_SPLIT"
    --config "$CONFIG_PATH"
    --dataset-id "$DATASET_ID"
  )
  if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
    INFER_CMD+=(--max-samples "$MAX_EVAL_SAMPLES")
  fi

  EVAL_CMD=(
    python -m scripts.evaluate_predictions
    --predictions-csv "${EVAL_DIR}/raw_predictions.csv"
    --output-dir "$EVAL_DIR"
    --cdm-toolkit-path "$CDM_TOOLKIT_PATH"
    --cdm-pools "$CDM_POOLS"
  )

  echo "[Inference Pipeline] Running split ${EVAL_SPLIT} with model ${MODEL_ID}"
  "${INFER_CMD[@]}"
  "${EVAL_CMD[@]}"
  EVAL_DIRS+=("${EVAL_DIR}")
  echo "[Inference Pipeline] Output: ${EVAL_DIR}"
done

OVERALL_DIR="${OUTPUT_DIR}/overall_results"
COLLECT_CMD=(
  python -m scripts.collect_eval_results
  --output-dir "${OVERALL_DIR}"
)
for EVAL_DIR in "${EVAL_DIRS[@]}"; do
  COLLECT_CMD+=(--eval-dir "${EVAL_DIR}")
done
"${COLLECT_CMD[@]}"

echo "[Inference Pipeline] Completed."
echo "[Inference Pipeline] Outputs root: ${OUTPUT_DIR}"
echo "[Inference Pipeline] Overall results: ${OVERALL_DIR}"
