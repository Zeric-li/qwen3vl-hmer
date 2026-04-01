from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.evaluate_predictions import summarize_buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-dir", dest="eval_dirs", action="append", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_split_rows(eval_dirs: list[Path]) -> tuple[list[dict[str, object]], pd.DataFrame]:
    split_rows: list[dict[str, object]] = []
    evaluated_frames: list[pd.DataFrame] = []

    for eval_dir in eval_dirs:
        metrics_path = eval_dir / "metrics.json"
        evaluated_path = eval_dir / "evaluated_predictions.csv"
        inference_summary_path = eval_dir / "inference_summary.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        if not evaluated_path.exists():
            raise FileNotFoundError(f"Missing evaluated predictions file: {evaluated_path}")
        if not inference_summary_path.exists():
            raise FileNotFoundError(f"Missing inference summary file: {inference_summary_path}")

        metrics = load_json(metrics_path)
        inference_summary = load_json(inference_summary_path)
        split_name = str(inference_summary["split"])

        split_rows.append(
            {
                "split": split_name,
                "eval_dir": str(eval_dir),
                "num_samples": int(metrics["num_samples"]),
                "exact_match_rate": float(metrics["primary_metrics"]["exact_match_rate"]),
                "math_verify_rate": float(metrics["primary_metrics"]["math_verify_rate"]),
                "cdm_exp_rate": float(metrics["primary_metrics"]["cdm_exp_rate"]),
                "avg_cer": float(metrics["text_metrics"]["avg_cer"]),
                "avg_edit_score": float(metrics["text_metrics"]["avg_edit_score"]),
                "avg_bleu4": float(metrics["text_metrics"]["avg_bleu4"]),
                "avg_cdm_f1": float(metrics["text_metrics"]["avg_cdm_f1"]),
                "avg_latency_s": float(metrics["runtime_metrics"]["avg_latency_s"]),
            }
        )

        evaluated_df = pd.read_csv(evaluated_path).copy()
        evaluated_df["eval_split"] = split_name
        evaluated_frames.append(evaluated_df)

    if not evaluated_frames:
        raise ValueError("No evaluation directories were provided.")

    return split_rows, pd.concat(evaluated_frames, ignore_index=True)


def build_overall_metrics(evaluated_df: pd.DataFrame, split_rows: list[dict[str, object]]) -> dict[str, object]:
    runtime_versions: dict[str, object] = {}
    if split_rows:
        first_metrics = load_json(Path(split_rows[0]["eval_dir"]) / "metrics.json")
        runtime_versions = first_metrics["runtime_metrics"].get("runtime_versions", {})

    return {
        "num_splits": len(split_rows),
        "num_samples": int(len(evaluated_df)),
        "evaluated_splits": [row["split"] for row in split_rows],
        "primary_metrics": {
            "exact_match_rate": float(evaluated_df["exact_match"].mean()),
            "math_verify_rate": float(evaluated_df["math_verify_match"].mean()),
            "cdm_exp_rate": float(evaluated_df["cdm_match"].mean()),
        },
        "text_metrics": {
            "avg_cer": float(evaluated_df["cer"].mean()),
            "avg_edit_score": float(evaluated_df["edit_score"].mean()),
            "avg_bleu4": float(evaluated_df["bleu4"].mean()),
            "avg_cdm_f1": float(evaluated_df["cdm_f1"].mean()),
        },
        "math_diagnostics": {
            "gold_parse_ok_rate": float(evaluated_df["math_verify_gold_parse_ok"].mean()),
            "pred_parse_ok_rate": float(evaluated_df["math_verify_pred_parse_ok"].mean()),
            "verify_attempt_rate": float(evaluated_df["math_verify_verify_attempted"].mean()),
            "timeout_rate": float(evaluated_df["math_verify_timeout"].mean()),
            "avg_parse_latency_ms": float(evaluated_df["math_verify_parse_latency_ms"].mean()),
            "avg_verify_latency_ms": float(evaluated_df["math_verify_verify_latency_ms"].mean()),
        },
        "runtime_metrics": {
            "avg_latency_s": float(evaluated_df["latency_s"].mean()),
            "runtime_versions": runtime_versions,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dirs = [Path(path) for path in args.eval_dirs]
    split_rows, evaluated_df = collect_split_rows(eval_dirs)

    split_summary_df = pd.DataFrame(split_rows).sort_values(by="split").reset_index(drop=True)
    overall_metrics = build_overall_metrics(evaluated_df, split_rows)
    bucket_rows = summarize_buckets(evaluated_df)

    split_summary_df.to_csv(output_dir / "split_metrics.csv", index=False)
    evaluated_df.to_csv(output_dir / "evaluated_predictions_all.csv", index=False)

    with open(output_dir / "split_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(split_rows, handle, indent=2)
    with open(output_dir / "overall_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(overall_metrics, handle, indent=2)
    with open(output_dir / "overall_bucket_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(bucket_rows, handle, indent=2)

    print(json.dumps(overall_metrics, indent=2))
    print(f"Saved overall results to: {output_dir}")


if __name__ == "__main__":
    main()
