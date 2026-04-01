from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd


def run_cdm_evaluation(
    rows: list[dict[str, object]],
    toolkit_path: str,
    work_dir: str,
    pools: int = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    input_path = work_path / "cdm_input.json"
    with open(input_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    evaluation_script = Path(toolkit_path) / "evaluation.py"
    if not evaluation_script.exists():
        evaluation_script = Path(toolkit_path) / "cdm" / "evaluation.py"
    if not evaluation_script.exists():
        raise FileNotFoundError(f"Cannot find UniMERNet CDM evaluation.py under: {toolkit_path}")

    command = [
        sys.executable,
        str(evaluation_script),
        "-i",
        str(input_path),
        "-o",
        str(work_path),
        "-p",
        str(pools),
    ]
    subprocess.run(command, check=True)

    result_dir = work_path / input_path.stem
    metrics_path = result_dir / "metrics_res.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"UniMERNet CDM did not produce metrics file: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    details = []
    for img_id, info in payload["details"].items():
        details.append(
            {
                "sample_id": img_id,
                "cdm_recall": info["recall"],
                "cdm_precision": info["precision"],
                "cdm_f1": info["F1_score"],
                "cdm_match": bool(math.isclose(float(info["F1_score"]), 1.0, rel_tol=0.0, abs_tol=1e-9)),
            }
        )

    metrics = {
        "cdm_mean_score": payload["mean_score"],
        "cdm_exp_rate": payload["exp_rate"],
    }
    return metrics, pd.DataFrame(details)
