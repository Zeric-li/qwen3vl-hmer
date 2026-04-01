from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OVERALL_COLOR = "#a8b5c0"


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate report figures. Install it in the active environment.") from exc

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_metrics = pd.read_csv(results_dir / "split_metrics.csv")
    evaluated_all = pd.read_csv(results_dir / "evaluated_predictions_all.csv")
    bucket_metrics = pd.read_json(results_dir / "overall_bucket_metrics.json")
    return split_metrics, evaluated_all, bucket_metrics


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _compute_axis_upper_bound(values: pd.Series, minimum: float = 1.0) -> float:
    if values.empty:
        return max(minimum, 1.0)

    max_value = float(values.max())
    if max_value <= 0:
        return max(minimum, 0.1)
    if max_value <= minimum:
        return minimum
    return max_value * 1.05


def _append_overall_split_metrics(split_metrics: pd.DataFrame) -> pd.DataFrame:
    total_samples = int(split_metrics["num_samples"].sum())
    weighted_columns = ["exact_match_rate", "avg_cer", "avg_edit_score", "avg_bleu4", "avg_latency_s"]

    overall_row = {
        "split": "Overall",
        "eval_dir": "overall_results",
        "num_samples": total_samples,
    }
    for column in weighted_columns:
        overall_row[column] = (
            float((split_metrics[column] * split_metrics["num_samples"]).sum()) / total_samples
            if total_samples > 0
            else 0.0
        )

    return pd.concat([split_metrics.copy(), pd.DataFrame([overall_row])], ignore_index=True)


def plot_split_metric_bars(split_metrics: pd.DataFrame, output_dir: Path) -> None:
    plt = _get_plt()
    split_metrics = _append_overall_split_metrics(split_metrics)
    metrics = [
        ("exact_match_rate", "Exact Match", "#0b6e4f"),
        ("avg_cer", "CER", "#c84c09"),
        ("avg_edit_score", "Edit Score", "#005f73"),
        ("avg_bleu4", "BLEU-4", "#7f5539"),
    ]
    x_positions = list(range(len(split_metrics)))
    split_labels = split_metrics["split"].astype(str).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Per-Split Metrics", fontweight="bold", y=0.98)

    for ax, (column, title, color) in zip(axes.flat, metrics):
        bar_colors = [color] * len(split_metrics)
        if bar_colors:
            bar_colors[-1] = OVERALL_COLOR
        ax.bar(x_positions, split_metrics[column], color=bar_colors, alpha=0.9, width=0.65)
        ax.set_title(title)
        ax.set_xlabel("Split")
        ax.set_xticks(x_positions, split_labels)
        upper_bound = (
            _compute_axis_upper_bound(split_metrics[column])
            if column == "avg_cer"
            else 1.0
        )
        ax.set_ylim(0.0, upper_bound)
        _style_axes(ax)
        for x, y in zip(x_positions, split_metrics[column]):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "per_split_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_error_bucket_profile(bucket_metrics: pd.DataFrame, output_dir: Path) -> None:
    plt = _get_plt()
    bucket_metrics = bucket_metrics.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(bucket_metrics["error_bucket"], bucket_metrics["count"], color=OVERALL_COLOR)
    ax.set_title("Error Bucket Profile", fontweight="bold")
    ax.set_xlabel("Count")
    _style_axes(ax)

    for y, value in zip(bucket_metrics["error_bucket"], bucket_metrics["count"]):
        ax.text(value, y, f" {int(value)}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "error_bucket_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_cer_distribution(evaluated_all: pd.DataFrame, output_dir: Path) -> None:
    plt = _get_plt()
    splits = list(dict.fromkeys(evaluated_all["eval_split"].tolist()))
    data = [evaluated_all.loc[evaluated_all["eval_split"] == split, "cer"].tolist() for split in splits]
    data.append(evaluated_all["cer"].tolist())
    labels = [str(split) for split in splits] + ["Overall"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#3f4e5a", "linewidth": 1.6},
    )
    palette = ["#0b6e4f", "#005f73", "#7f5539", "#9b2226"]
    box_colors = palette[: len(bp["boxes"])]
    if box_colors:
        box_colors[-1] = OVERALL_COLOR
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_title("CER Distribution by Split", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("CER")
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_dir / "cer_distribution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "report_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics, evaluated_all, bucket_metrics = _load_inputs(results_dir)
    plot_split_metric_bars(split_metrics, output_dir)
    plot_error_bucket_profile(bucket_metrics, output_dir)
    plot_cer_distribution(evaluated_all, output_dir)

    print(f"Saved report figures to: {output_dir}")


if __name__ == "__main__":
    main()
