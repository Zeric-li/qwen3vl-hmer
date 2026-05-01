# Language-Side LoRA Adaptation of Qwen3-VL for Handwritten Mathematical Expression Recognition

A Python project for supervised LoRA fine-tuning of Qwen VL models on handwritten mathematical expression (HME) transcription.

## Install

```bash
pip install -r requirements.txt
```

If `transformers` support for Qwen3-VL lags in your environment, install the latest version from source:

```bash
pip install -U "git+https://github.com/huggingface/transformers"
```

## Colab

Use [notebooks/crohme_colab_quickstart.ipynb](notebooks/crohme_colab_quickstart.ipynb) to run the full Colab demo in a GPU runtime.

Colab demo flow:

1. Open the notebook in a GPU runtime.
2. Optionally adjust the sample caps and output root in the parameter cell.
3. Run all cells.

The notebook clones `main`, installs the dependencies, writes Colab-safe configs for both pipelines, runs the baseline inference pipeline and the LoRA train+eval pipeline, then generates visualizations and a side-by-side comparison of the aggregated results. The default Colab config cell is aligned with the current pipeline YAML defaults and can be edited directly in the notebook.

## Train

```bash
python -m scripts.train_lora \
  --config configs/crohme_lora_pipeline.yaml
```

## Inference

```bash
python -m scripts.run_inference \
  --checkpoint ./outputs/qwen3vl-crohme-lora/checkpoint-final \
  --config configs/crohme_lora_pipeline.yaml \
  --split 2019
```

This writes `raw_predictions.csv` with the raw model text, raw gold labels,
and per-sample latency. Normalization happens during the evaluation step.

## Evaluate

Full pipelines:

```bash
./scripts/run_inference_pipeline.sh configs/crohme_inference_pipeline.yaml
./scripts/run_lora_pipeline.sh configs/crohme_lora_pipeline.yaml
```

Smoke pipelines:

```bash
./scripts/run_inference_pipeline.sh configs/crohme_inference_smoke.yaml
./scripts/run_lora_pipeline.sh configs/crohme_lora_smoke.yaml
```

The full configs run the complete benchmark setup. The smoke configs reduce the run to a single split with a small sample cap and write outputs under `*-smoke`.
The training config can set `train_eval_split` for the single split used by in-training validation, while `eval_splits` controls the post-training benchmark runs.

If `./outputs/qwen3vl-crohme-lora/checkpoint-final/adapter_config.json` already exists, the LoRA pipeline skips training and goes straight to evaluation.

After all configured splits finish, the pipeline also writes an aggregated `overall_results/` directory with:

- `overall_metrics.json`
- `split_metrics.json`
- `split_metrics.csv`
- `evaluated_predictions_all.csv`
- `overall_bucket_metrics.json`

The pipeline now also generates single-experiment PNG figures automatically under `overall_results/report_figures/`.
If you want to regenerate them manually, use:

```bash
python -m scripts.generate_eval_report_figures \
  --results-dir ./outputs/qwen3vl-crohme-base/overall_results
```

This writes PNG figures under `overall_results/report_figures/`.

To regenerate both the per-experiment figures and the cross-experiment comparison figures in one step, use:

```bash
./scripts/run_report_visualization.sh \
  base=./outputs/qwen3vl-crohme-base/overall_results \
  lora=./outputs/qwen3vl-crohme-lora/overall_results \
  --reference-label base
```

This writes:

- per-experiment figures under each input `overall_results/report_figures/`
- shared comparison figures under `outputs/experiment_comparisons/`
- pairwise sample-level figures for each non-reference experiment under `outputs/experiment_comparisons/vs_<label>/`

The comparison outputs currently include:

- `overall_metric_comparison.png`
- `per_split_metric_comparison.png`
- `latency_comparison.png`
- `vs_<label>/outcome_transition_comparison.png`
- `vs_<label>/cer_improvement_distribution.png`
- `vs_<label>/error_bucket_shift.png`

## Evaluate A Single Split

```bash
python -m scripts.evaluate_predictions \
  --predictions-csv ./outputs/qwen3vl-crohme-lora/checkpoint-final/eval_2019/raw_predictions.csv
```

This writes:

- `evaluated_predictions.csv`
- `metrics.json`
- `bucket_metrics.json`
- `error_samples.csv`
- `unimernet_cdm_input.json`

The evaluation stack is aligned to CROHME-style tokenized LaTeX and matches the local text-based metrics described in Uni-MuMER. It reports:

- exact match on normalized LaTeX
- CER
- edit score
- BLEU-4

## External CDM

If you want `ExpRate@CDM`, export the predictions and run the official UniMERNet evaluator in its own environment.

`scripts.evaluate_predictions` now writes `unimernet_cdm_input.json` automatically. You can also export it manually:

```bash
python -m scripts.export_unimernet_cdm_input \
  --predictions-csv ./outputs/qwen3vl-crohme-lora/checkpoint-final/eval_2019/raw_predictions.csv
```

This writes `unimernet_cdm_input.json` with the `img_id`, `gt`, and `pred` fields expected by the official toolkit.

For the external CDM step, use the official UniMERNet evaluator and its setup guide:

- https://github.com/opendatalab/UniMERNet/tree/main/cdm

## Error Analysis

```bash
python -m scripts.analyze_errors \
  --evaluated-csv ./outputs/qwen3vl-crohme-lora/checkpoint-final/eval_2019/evaluated_predictions.csv
```

## Dataset assumptions

The code expects a Hugging Face dataset with these fields:

- `image`: PIL image or image-like object accepted by `datasets`
- `label`: ground-truth LaTeX string

This matches `Neeze/CROHME-full` for the benchmark splits `2014`, `2016`, and `2019`.

## Notes

- The training objective masks prompt tokens and only computes loss on the assistant answer tokens.
- The default LoRA target list focuses on the language model attention projections.
- The vision backbone stays frozen in the MVP.
