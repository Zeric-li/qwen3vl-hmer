from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm.auto import tqdm

from scripts.export_unimernet_cdm_input import build_unimernet_cdm_rows
from hme_vlm.evaluators.text import bleu4, character_error_rate, edit_score, exact_match
from hme_vlm.normalization import canonicalize_latex_for_metrics, clean_model_text, normalize_crohme_latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def assign_error_bucket(row: pd.Series) -> str:
    gold = row["gold_latex_normalized"]
    pred = row["pred_latex_normalized"]

    if not pred:
        return "empty_prediction"
    if row["exact_match"]:
        return "exact_match"
    if len(gold) < 12:
        return "short_symbol_confusion"
    if any(token in gold for token in [r"\sum", r"\int", r"\lim"]):
        return "sum_integral_limit"
    if any(token in gold for token in [r"\frac", r"\sqrt"]):
        return "fraction_root"
    if any(token in pred for token in [r"\alpha", r"\beta", r"\gamma", r"\mathbb"]):
        return "greek_or_font_confusion"
    if "^" in gold or "_" in gold:
        return "dense_script_structure"
    if len(gold) >= 40:
        return "long_expression"
    return "general_transcription_error"


def summarize_buckets(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket, group in df.groupby("error_bucket", dropna=False):
        rows.append(
            {
                "error_bucket": bucket,
                "count": int(len(group)),
                "exact_match_rate": float(group["exact_match"].mean()),
                "avg_cer": float(group["cer"].mean()),
                "avg_bleu4": float(group["bleu4"].mean()),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def main() -> None:
    args = parse_args()
    from pathlib import Path

    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    required_columns = {"sample_id", "source", "gold_latex_raw", "pred_text_raw", "latency_s"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in raw predictions csv: {sorted(missing_columns)}")

    df["gold_latex_normalized"] = df["gold_latex_raw"].fillna("").astype(str).map(normalize_crohme_latex)
    df["pred_latex_raw"] = df["pred_text_raw"].fillna("").astype(str).map(clean_model_text)
    df["pred_latex_normalized"] = df["pred_latex_raw"].map(normalize_crohme_latex)
    df["gold_latex_canonical"] = df["gold_latex_normalized"].map(canonicalize_latex_for_metrics)
    df["pred_latex_canonical"] = df["pred_latex_normalized"].map(canonicalize_latex_for_metrics)

    rows: list[dict[str, object]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluate predictions"):
        exact = exact_match(row.gold_latex_normalized, row.pred_latex_normalized)

        updated = {
            "sample_id": row.sample_id,
            "source": row.source,
            "gold_latex_raw": row.gold_latex_raw,
            "pred_text_raw": row.pred_text_raw,
            "gold_latex_normalized": row.gold_latex_normalized,
            "pred_latex_raw": row.pred_latex_raw,
            "pred_latex_normalized": row.pred_latex_normalized,
            "exact_match": exact,
            "cer": character_error_rate(row.gold_latex_normalized, row.pred_latex_normalized),
            "edit_score": edit_score(row.gold_latex_normalized, row.pred_latex_normalized),
            "bleu4": bleu4(row.gold_latex_normalized, row.pred_latex_normalized),
            "latency_s": row.latency_s,
        }
        rows.append(updated)

    evaluated_df = pd.DataFrame(rows)
    evaluated_df["error_bucket"] = evaluated_df.apply(assign_error_bucket, axis=1)
    evaluated_df.to_csv(output_dir / "evaluated_predictions.csv", index=False)

    metrics = {
        "num_samples": int(len(evaluated_df)),
        "exact_match_rate": float(evaluated_df["exact_match"].mean()),
        "avg_cer": float(evaluated_df["cer"].mean()),
        "avg_edit_score": float(evaluated_df["edit_score"].mean()),
        "avg_bleu4": float(evaluated_df["bleu4"].mean()),
        "avg_latency_s": float(evaluated_df["latency_s"].mean()),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    bucket_rows = summarize_buckets(evaluated_df)
    with open(output_dir / "bucket_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(bucket_rows, handle, indent=2)

    error_samples = evaluated_df[evaluated_df["exact_match"] == False].copy()
    error_samples.to_csv(output_dir / "error_samples.csv", index=False)

    unimernet_rows = build_unimernet_cdm_rows(df)
    with open(output_dir / "unimernet_cdm_input.json", "w", encoding="utf-8") as handle:
        json.dump(unimernet_rows, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluated predictions to: {output_dir / 'evaluated_predictions.csv'}")


if __name__ == "__main__":
    main()
