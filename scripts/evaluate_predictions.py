from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from hme_vlm.evaluators.math_verify import MathVerifier, MathVerifyConfig
from hme_vlm.evaluators.text import bleu4, character_error_rate, edit_score, exact_match
from hme_vlm.normalization import canonicalize_latex_for_metrics, clean_model_text, normalize_crohme_latex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--math-verify-parsing-timeout-seconds", type=int, default=2)
    parser.add_argument("--math-verify-timeout-seconds", type=int, default=2)
    parser.add_argument("--math-verify-no-expr-fallback", action="store_true")
    parser.add_argument("--cdm-toolkit-path", type=str, default="./external/UniMERNet/cdm")
    parser.add_argument("--cdm-pools", type=int, default=1)
    return parser.parse_args()


def assign_error_bucket(row: pd.Series) -> str:
    gold = row["gold_latex_normalized"]
    pred = row["pred_latex_normalized"]

    if not pred:
        return "empty_prediction"
    if row["exact_match"]:
        return "exact_match"
    if row["math_verify_match"]:
        return "exact_miss_math_verify_hit"
    if not row["math_verify_gold_parse_ok"] or not row["math_verify_pred_parse_ok"]:
        return "math_verify_parse_failure"
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
                "math_verify_rate": float(group["math_verify_match"].mean()),
                "avg_cer": float(group["cer"].mean()),
                "avg_bleu4": float(group["bleu4"].mean()),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def run_cdm_evaluation_required(
    df: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    from hme_vlm.evaluators.cdm import run_cdm_evaluation

    rows = [
        {
            "img_id": row.sample_id,
            "gt": row.gold_latex_normalized,
            "pred": row.pred_latex_normalized,
        }
        for row in df.itertuples(index=False)
    ]
    metrics, details = run_cdm_evaluation(
        rows=rows,
        toolkit_path=args.cdm_toolkit_path,
        work_dir=str(output_dir / "cdm"),
        pools=args.cdm_pools,
    )

    expected_sample_ids = set(df["sample_id"].tolist())
    observed_sample_ids = set(details["sample_id"].tolist())
    if observed_sample_ids != expected_sample_ids:
        missing_ids = sorted(expected_sample_ids - observed_sample_ids)
        extra_ids = sorted(observed_sample_ids - expected_sample_ids)
        raise ValueError(
            "UniMERNet CDM output sample coverage mismatch. "
            f"missing={missing_ids[:5]} extra={extra_ids[:5]}"
        )

    merged = df.merge(details, on="sample_id", how="left")
    with open(output_dir / "cdm_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return merged, metrics


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cdm_toolkit_path = Path(args.cdm_toolkit_path)
    if not cdm_toolkit_path.exists():
        raise FileNotFoundError(
            "UniMERNet CDM is the default evaluator but the toolkit path was not found: "
            f"{cdm_toolkit_path}. Clone UniMERNet under ./external/UniMERNet and install its requirements first."
        )

    df = pd.read_csv(predictions_csv)
    required_columns = {"sample_id", "source", "gold_latex_raw", "pred_text_raw", "latency_s"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in raw predictions csv: {sorted(missing_columns)}")

    verifier = MathVerifier(
        MathVerifyConfig(
            parsing_timeout_seconds=args.math_verify_parsing_timeout_seconds,
            verify_timeout_seconds=args.math_verify_timeout_seconds,
            pred_use_expr_fallback=not args.math_verify_no_expr_fallback,
        )
    )

    df["gold_latex_normalized"] = df["gold_latex_raw"].fillna("").astype(str).map(normalize_crohme_latex)
    df["pred_latex_raw"] = df["pred_text_raw"].fillna("").astype(str).map(clean_model_text)
    df["pred_latex_normalized"] = df["pred_latex_raw"].map(normalize_crohme_latex)
    df["gold_latex_canonical"] = df["gold_latex_normalized"].map(canonicalize_latex_for_metrics)
    df["pred_latex_canonical"] = df["pred_latex_normalized"].map(canonicalize_latex_for_metrics)

    verifier.preparse_gold(df["gold_latex_normalized"].tolist())

    rows: list[dict[str, object]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluate predictions"):
        exact = exact_match(row.gold_latex_normalized, row.pred_latex_normalized)
        math_verify_result = verifier.match(
            gold_latex=row.gold_latex_normalized,
            pred_latex=row.pred_latex_normalized,
        )

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
            **math_verify_result.as_row_fields(),
            "latency_s": row.latency_s,
        }
        rows.append(updated)

    evaluated_df = pd.DataFrame(rows)
    evaluated_df["error_bucket"] = evaluated_df.apply(assign_error_bucket, axis=1)
    evaluated_df, cdm_metrics = run_cdm_evaluation_required(evaluated_df, args, output_dir)
    evaluated_df.to_csv(output_dir / "evaluated_predictions.csv", index=False)

    metrics = {
        "num_samples": int(len(evaluated_df)),
        "primary_metrics": {
            "exact_match_rate": float(evaluated_df["exact_match"].mean()),
            "math_verify_rate": float(evaluated_df["math_verify_match"].mean()),
            "cdm_exp_rate": float(cdm_metrics["cdm_exp_rate"]),
        },
        "text_metrics": {
            "avg_cer": float(evaluated_df["cer"].mean()),
            "avg_edit_score": float(evaluated_df["edit_score"].mean()),
            "avg_bleu4": float(evaluated_df["bleu4"].mean()),
            "avg_cdm_f1": float(cdm_metrics["cdm_mean_score"]),
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
            "runtime_versions": verifier.runtime_versions,
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    bucket_rows = summarize_buckets(evaluated_df)
    with open(output_dir / "bucket_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(bucket_rows, handle, indent=2)

    error_samples = evaluated_df[evaluated_df["exact_match"] == False].copy()
    error_samples.to_csv(output_dir / "error_samples.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluated predictions to: {output_dir / 'evaluated_predictions.csv'}")


if __name__ == "__main__":
    main()
