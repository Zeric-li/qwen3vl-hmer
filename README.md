# Qwen3-VL HME LoRA MVP

A minimal pure-Python project for supervised LoRA fine-tuning of `Qwen/Qwen3-VL-4B-Instruct` on handwritten mathematical expression (HME) transcription.

## Scope

This project is intentionally small and direct:

- one training script
- one inference script
- one evaluation script
- one error-analysis script
- one simple dataset adapter
- explicit normalization and evaluator modules

The default dataset target is `Neeze/CROHME-full` from Hugging Face.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
git clone https://github.com/opendatalab/UniMERNet.git external/UniMERNet
pip install -r external/UniMERNet/cdm/requirements.txt
```

If `transformers` support for Qwen2.5-VL lags in your environment, install the latest version from source:

```bash
pip install -U "git+https://github.com/huggingface/transformers"
```

## Colab

Use [crohme_colab_quickstart.ipynb](/home/zeric/projects/comp646/HME/notebooks/crohme_colab_quickstart.ipynb) to run either pipeline in Google Colab with a GPU runtime.

Minimal Colab flow:

1. Open the notebook in a GPU runtime.
2. Set `REPO_URL`.
3. Choose `TASK = "inference_only"` or `TASK = "lora_train_eval"`.
4. Run all cells.

The notebook installs the required system tools for UniMERNet CDM, clones `external/UniMERNet`, writes `configs/colab_<task>.yaml`, and runs the matching pipeline script under `scripts/`.

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

```bash
./scripts/run_inference_pipeline.sh configs/crohme_inference_pipeline.yaml
```

This runs direct base-model inference and evaluation on the `2014`, `2016`, and `2019` test splits.

```bash
./scripts/run_lora_pipeline.sh configs/crohme_lora_pipeline.yaml
```

This trains on the `train` split, then evaluates the final LoRA checkpoint on `2014`, `2016`, and `2019`.
If `./outputs/qwen3vl-crohme-lora/checkpoint-final/adapter_config.json` already exists, training is skipped and the pipeline goes straight to evaluation.
The training config can set `train_eval_split` for the single split used by in-training validation, while `eval_splits` controls the post-training benchmark runs.
Both pipeline configs support optional UniMERNet CDM fields:

- `cdm_toolkit_path`
- `cdm_pools`

After all configured splits finish, the pipeline also writes an aggregated `overall_results/` directory with:

- `overall_metrics.json`
- `split_metrics.json`
- `split_metrics.csv`
- `evaluated_predictions_all.csv`
- `overall_bucket_metrics.json`

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

The evaluation stack is aligned to CROHME-style tokenized LaTeX and uses UniMERNet CDM as the default expression-rate backend. It reports:

- exact match on normalized LaTeX
- CER
- edit score
- BLEU-4
- Math-Verify diagnostics
- ExpRate@CDM through the UniMERNet evaluator

The default expected evaluator layout is:

```bash
./external/UniMERNet/cdm
```

### Configure UniMERNet CDM

The official UniMERNet CDM evaluator is the default evaluation backend in this project. It is not a pure Python package. It renders LaTeX to files and then performs image-level matching, so it needs extra system tools in addition to Python dependencies.

Recommended local layout:

```bash
git clone https://github.com/opendatalab/UniMERNet.git external/UniMERNet
```

Expected evaluator path in this project:

```bash
./external/UniMERNet/cdm
```

System dependencies required by the official README:

```bash
node -v
convert --version
pdflatex --version
```

If any of those commands are missing, install:

- Node.js
- ImageMagick
- a LaTeX distribution with `pdflatex` available, for example `texlive-full` on Ubuntu

Install the CDM Python-side dependencies inside your current environment:

```bash
pip install -r external/UniMERNet/cdm/requirements.txt
```

At the time of writing, the upstream CDM requirements file includes `tqdm`, `matplotlib`, `numpy<2.0.0`, `scikit-image<=0.20.0`, `opencv-python`, and optional `gradio==4.43.0`. This is separate from this repo's main `requirements.txt`.

Minimal smoke check for the evaluator itself:

```bash
python external/UniMERNet/cdm/evaluation.py --help
```

Then run this project's evaluator:

```bash
python -m scripts.evaluate_predictions \
  --predictions-csv ./outputs/qwen3vl-crohme-lora/checkpoint-final/eval_2019/raw_predictions.csv \
  --output-dir ./outputs/qwen3vl-crohme-lora/checkpoint-final/eval_2019 \
  --cdm-toolkit-path ./external/UniMERNet/cdm \
  --cdm-pools 8
```

This project writes the UniMERNet-compatible CDM input JSON for you and reads back `metrics_res.json` from the official evaluator output directory.

The config-driven pipelines now require `cdm_toolkit_path` and default it to `./external/UniMERNet/cdm` in both [crohme_inference_pipeline.yaml](/home/zeric/projects/comp646/HME/configs/crohme_inference_pipeline.yaml) and [crohme_lora_pipeline.yaml](/home/zeric/projects/comp646/HME/configs/crohme_lora_pipeline.yaml).

Follow the upstream setup guide if you need the full installation details or Docker-based setup:

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
- Math-Verify is used as a diagnostic evaluator, not as the primary CROHME metric.
