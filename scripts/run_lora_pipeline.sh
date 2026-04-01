#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_lora_pipeline.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

readarray -t CONFIG_VALUES < <(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config

config = load_yaml_config(sys.argv[1])
if config.get("pipeline_type") != "lora_train_eval":
    raise SystemExit(f"Expected pipeline_type=lora_train_eval, got {config.get('pipeline_type')}")

print(config["output_dir"])
print(config.get("eval_dataset_id", config.get("train_dataset_id", "Neeze/CROHME-full")))
print("" if config.get("max_eval_samples") is None else config["max_eval_samples"])
for split in config.get("eval_splits", []):
    print(f"SPLIT::{split}")
PY
)

OUTPUT_DIR="${CONFIG_VALUES[0]}"
DATASET_ID="${CONFIG_VALUES[1]}"
MAX_EVAL_SAMPLES="${CONFIG_VALUES[2]}"

EVAL_SPLITS=()
EVAL_DIRS=()
for value in "${CONFIG_VALUES[@]:3}"; do
  EVAL_SPLITS+=("${value#SPLIT::}")
done

if [[ "${#EVAL_SPLITS[@]}" -eq 0 ]]; then
  echo "Config must define at least one eval_splits entry: ${CONFIG_PATH}" >&2
  exit 1
fi

TRAINED_CHECKPOINT="${OUTPUT_DIR}/checkpoint-final"
if [[ -f "${TRAINED_CHECKPOINT}/adapter_config.json" ]]; then
  echo "[LoRA Pipeline] Found existing checkpoint, skipping training: ${TRAINED_CHECKPOINT}"
else
  echo "[LoRA Pipeline] Training with config: ${CONFIG_PATH}"
  python -m scripts.train_lora --config "$CONFIG_PATH"
fi

for EVAL_SPLIT in "${EVAL_SPLITS[@]}"; do
  EVAL_DIR="${TRAINED_CHECKPOINT}/eval_${EVAL_SPLIT}"

  INFER_CMD=(
    python -m scripts.run_inference
    --checkpoint "$TRAINED_CHECKPOINT"
    --output-dir "$EVAL_DIR"
    --batch-size "${FT_EVAL_BATCH_SIZE:-1}"
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
  )

  echo "[LoRA Pipeline] Running split ${EVAL_SPLIT} with checkpoint ${TRAINED_CHECKPOINT}"
  "${INFER_CMD[@]}"
  "${EVAL_CMD[@]}"
  EVAL_DIRS+=("${EVAL_DIR}")
  echo "[LoRA Pipeline] Output: ${EVAL_DIR}"
done

OVERALL_DIR="${TRAINED_CHECKPOINT}/overall_results"
COLLECT_CMD=(
  python -m scripts.collect_eval_results
  --output-dir "${OVERALL_DIR}"
)
for EVAL_DIR in "${EVAL_DIRS[@]}"; do
  COLLECT_CMD+=(--eval-dir "${EVAL_DIR}")
done
"${COLLECT_CMD[@]}"

echo "[LoRA Pipeline] Completed."
echo "[LoRA Pipeline] Checkpoint: ${TRAINED_CHECKPOINT}"
echo "[LoRA Pipeline] Overall results: ${OVERALL_DIR}"
