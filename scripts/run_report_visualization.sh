#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ "$#" -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./scripts/run_report_visualization.sh label=/path/to/overall_results [label=/path/to/overall_results ...] [--reference-label LABEL] [--output-dir DIR]

Examples:
  ./scripts/run_report_visualization.sh \
    base=./outputs/qwen3vl-crohme-base/overall_results \
    lora=./outputs/qwen3vl-crohme-lora/overall_results
EOF
  exit 1
fi

REFERENCE_LABEL=""
OUTPUT_DIR=""
EXPERIMENT_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --reference-label)
      REFERENCE_LABEL="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    *)
      EXPERIMENT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#EXPERIMENT_ARGS[@]}" -lt 1 ]]; then
  echo "At least one experiment must be provided." >&2
  exit 1
fi

REFERENCE_LABEL="${REFERENCE_LABEL:-${EXPERIMENT_ARGS[0]%%=*}}"

COMPARISON_CMD=(
  python -m scripts.generate_experiment_comparison_figures
)
if [[ -n "$OUTPUT_DIR" ]]; then
  COMPARISON_CMD+=(--output-dir "$OUTPUT_DIR")
fi
COMPARISON_CMD+=(--reference-label "$REFERENCE_LABEL")

for EXPERIMENT_DEF in "${EXPERIMENT_ARGS[@]}"; do
  if [[ "$EXPERIMENT_DEF" != *=* ]]; then
    echo "Invalid experiment definition: $EXPERIMENT_DEF" >&2
    exit 1
  fi
  LABEL="${EXPERIMENT_DEF%%=*}"
  RESULTS_DIR="${EXPERIMENT_DEF#*=}"
  echo "[Report Viz] Generating single-experiment figures for ${LABEL}: ${RESULTS_DIR}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    python -m scripts.generate_eval_report_figures --results-dir "$RESULTS_DIR"
  COMPARISON_CMD+=(--experiment "$EXPERIMENT_DEF")
done

if [[ "${#EXPERIMENT_ARGS[@]}" -ge 2 ]]; then
  echo "[Report Viz] Generating cross-experiment comparison figures (reference=${REFERENCE_LABEL})"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" "${COMPARISON_CMD[@]}"
fi

echo "[Report Viz] Completed."
