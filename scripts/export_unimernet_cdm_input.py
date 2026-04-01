from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hme_vlm.normalization import clean_model_text, normalize_crohme_latex


def build_unimernet_cdm_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    required_columns = {"sample_id", "gold_latex_raw", "pred_text_raw"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in raw predictions csv: {sorted(missing_columns)}")

    return [
        {
            "img_id": str(row.sample_id),
            "gt": normalize_crohme_latex(str(row.gold_latex_raw or "")),
            "pred": normalize_crohme_latex(clean_model_text(str(row.pred_text_raw or ""))),
        }
        for row in df.itertuples(index=False)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    output_json = Path(args.output_json) if args.output_json else predictions_csv.with_name("unimernet_cdm_input.json")

    df = pd.read_csv(predictions_csv)
    rows = build_unimernet_cdm_rows(df)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(f"Saved UniMERNet CDM input to: {output_json}")


if __name__ == "__main__":
    main()
