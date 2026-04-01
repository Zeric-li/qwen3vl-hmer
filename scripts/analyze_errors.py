from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluated_csv = Path(args.evaluated_csv)
    output_dir = Path(args.output_dir) if args.output_dir else evaluated_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(evaluated_csv)
    if "error_bucket" not in df.columns:
        raise ValueError("evaluated csv must contain error_bucket column")

    summary_rows: list[dict[str, object]] = []
    top_examples: list[pd.DataFrame] = []

    for bucket, group in df.groupby("error_bucket", dropna=False):
        summary_rows.append(
            {
                "error_bucket": bucket,
                "count": int(len(group)),
                "exact_match_rate": float(group["exact_match"].mean()),
                "math_verify_rate": float(group["math_verify_match"].mean()),
                "avg_cer": float(group["cer"].mean()),
                "avg_bleu4": float(group["bleu4"].mean()),
            }
        )

        sample = group.sort_values(["cer", "bleu4"], ascending=[False, True]).head(args.topk).copy()
        sample.insert(0, "bucket", bucket)
        top_examples.append(sample)

    summary_rows.sort(key=lambda row: row["count"], reverse=True)
    with open(output_dir / "error_bucket_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    if top_examples:
        pd.concat(top_examples, ignore_index=True).to_csv(output_dir / "error_bucket_examples.csv", index=False)

    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
