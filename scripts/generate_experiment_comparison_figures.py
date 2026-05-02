from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

QUALITY_METRICS = [
    ("exact_match_rate", "Exact Match ↑", True),
    ("avg_cer", "CER ↓", False),
    ("avg_edit_score", "Edit Score ↑", True),
    ("avg_bleu4", "BLEU-4 ↑", True),
]
OUTCOME_ORDER = ["both_correct", "fixed_by_candidate", "regressed_after_candidate", "both_wrong"]
OUTCOME_LABELS = {
    "both_correct": "Both Correct",
    "fixed_by_candidate": "Fixed by Candidate",
    "regressed_after_candidate": "Regressed After Candidate",
    "both_wrong": "Both Wrong",
}
OUTCOME_COLORS = {
    "both_correct": "#4c956c",
    "fixed_by_candidate": "#2a9d8f",
    "regressed_after_candidate": "#bc4749",
    "both_wrong": "#b8c0c8",
}
FIGURE_FORMATS = ("png", "pdf", "svg")
DISPLAY_LABELS = {
    "base": "Base Qwen3-VL",
    "lora": "Adapted Qwen3-VL",
}


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate comparison figures. Install it in the active environment.") from exc

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    for figure_format in FIGURE_FORMATS:
        fig.savefig(output_dir / f"{stem}.{figure_format}", bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        required=True,
        help="Experiment definition in the form label=/path/to/overall_results. Pass at least two times.",
    )
    parser.add_argument(
        "--reference-label",
        type=str,
        default=None,
        help="Reference experiment label used for pairwise sample-level comparisons. Defaults to the first --experiment label.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def parse_experiment_args(experiment_args: list[str]) -> list[tuple[str, Path]]:
    if len(experiment_args) < 2:
        raise ValueError("At least two --experiment arguments are required.")

    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in experiment_args:
        if "=" not in item:
            raise ValueError(f"Invalid experiment definition: {item}. Expected label=/path/to/overall_results.")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Experiment label cannot be empty: {item}")
        if label in seen_labels:
            raise ValueError(f"Experiment labels must be unique: {label}")
        path = Path(raw_path).expanduser()
        parsed.append((label, path))
        seen_labels.add(label)
    return parsed


def load_json(path: Path) -> dict | list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment_results(label: str, results_dir: Path) -> dict[str, object]:
    split_metrics = pd.read_csv(results_dir / "split_metrics.csv").copy()
    evaluated_all = pd.read_csv(results_dir / "evaluated_predictions_all.csv").copy()
    overall_metrics = load_json(results_dir / "overall_metrics.json")
    bucket_metrics = pd.DataFrame(load_json(results_dir / "overall_bucket_metrics.json"))

    split_metrics["split"] = split_metrics["split"].astype(str)
    evaluated_all["eval_split"] = evaluated_all["eval_split"].astype(str)

    return {
        "label": label,
        "results_dir": results_dir,
        "split_metrics": split_metrics,
        "evaluated_all": evaluated_all,
        "overall_metrics": overall_metrics,
        "bucket_metrics": bucket_metrics,
    }


def build_metric_table(experiments: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        split_metrics = experiment["split_metrics"].copy()
        split_metrics["experiment"] = experiment["label"]
        rows.extend(split_metrics.to_dict(orient="records"))

        overall_metrics = experiment["overall_metrics"]
        rows.append(
            {
                "experiment": experiment["label"],
                "split": "Overall",
                "eval_dir": str(experiment["results_dir"]),
                "num_samples": int(overall_metrics["num_samples"]),
                "exact_match_rate": float(overall_metrics["exact_match_rate"]),
                "avg_cer": float(overall_metrics["avg_cer"]),
                "avg_edit_score": float(overall_metrics["avg_edit_score"]),
                "avg_bleu4": float(overall_metrics["avg_bleu4"]),
                "avg_latency_s": float(overall_metrics["avg_latency_s"]),
            }
        )

    metric_table = pd.DataFrame(rows)
    split_order = [split for split in metric_table["split"].tolist() if split != "Overall"]
    split_order = sorted(dict.fromkeys(split_order), key=lambda item: (item == "Overall", item))
    split_order.append("Overall")
    metric_table["split"] = pd.Categorical(metric_table["split"], categories=split_order, ordered=True)
    return metric_table.sort_values(["split", "experiment"]).reset_index(drop=True)


def build_color_map(labels: list[str]) -> dict[str, str]:
    palette = [
        "#577590",
        "#43aa8b",
        "#bc4749",
        "#f8961e",
        "#6d597a",
        "#277da1",
        "#90be6d",
        "#f9844a",
    ]
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def build_pairwise_comparison(reference_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    merged = reference_df.merge(
        candidate_df,
        on=["sample_id", "eval_split"],
        suffixes=("_reference", "_candidate"),
        how="inner",
    )
    if len(merged) != len(reference_df) or len(merged) != len(candidate_df):
        raise ValueError("The two experiments do not contain the same evaluated samples.")

    merged["cer_improvement"] = merged["cer_reference"] - merged["cer_candidate"]
    merged["exact_delta"] = merged["exact_match_candidate"].astype(int) - merged["exact_match_reference"].astype(int)
    merged["outcome"] = "both_wrong"
    merged.loc[
        (merged["exact_match_reference"] == True) & (merged["exact_match_candidate"] == True),
        "outcome",
    ] = "both_correct"
    merged.loc[
        (merged["exact_match_reference"] == False) & (merged["exact_match_candidate"] == True),
        "outcome",
    ] = "fixed_by_candidate"
    merged.loc[
        (merged["exact_match_reference"] == True) & (merged["exact_match_candidate"] == False),
        "outcome",
    ] = "regressed_after_candidate"
    return merged


def build_pairwise_summary(pairwise_df: pd.DataFrame, reference_label: str, candidate_label: str) -> dict[str, object]:
    summary = {
        "reference_experiment": reference_label,
        "candidate_experiment": candidate_label,
        "num_samples": int(len(pairwise_df)),
        "fixed_by_candidate": int((pairwise_df["outcome"] == "fixed_by_candidate").sum()),
        "regressed_after_candidate": int((pairwise_df["outcome"] == "regressed_after_candidate").sum()),
        "both_correct": int((pairwise_df["outcome"] == "both_correct").sum()),
        "both_wrong": int((pairwise_df["outcome"] == "both_wrong").sum()),
        "avg_cer_improvement": float(pairwise_df["cer_improvement"].mean()),
        "median_cer_improvement": float(pairwise_df["cer_improvement"].median()),
        "candidate_win_rate_on_cer": float((pairwise_df["cer_improvement"] > 0).mean()),
        "candidate_loss_rate_on_cer": float((pairwise_df["cer_improvement"] < 0).mean()),
    }
    return summary


def build_outcome_table(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in pairwise_df.groupby("eval_split", sort=True):
        total = len(group)
        row = {"split": str(split), "num_samples": int(total)}
        for outcome in OUTCOME_ORDER:
            count = int((group["outcome"] == outcome).sum())
            row[outcome] = count
            row[f"{outcome}_rate"] = float(count / total) if total else 0.0
        rows.append(row)

    total = len(pairwise_df)
    overall_row = {"split": "Overall", "num_samples": int(total)}
    for outcome in OUTCOME_ORDER:
        count = int((pairwise_df["outcome"] == outcome).sum())
        overall_row[outcome] = count
        overall_row[f"{outcome}_rate"] = float(count / total) if total else 0.0
    rows.append(overall_row)
    return pd.DataFrame(rows)


def build_bucket_shift_table(
    reference_bucket_metrics: pd.DataFrame,
    candidate_bucket_metrics: pd.DataFrame,
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    ref = reference_bucket_metrics[["error_bucket", "count"]].rename(columns={"count": f"count_{reference_label}"})
    cand = candidate_bucket_metrics[["error_bucket", "count"]].rename(columns={"count": f"count_{candidate_label}"})
    merged = ref.merge(cand, on="error_bucket", how="outer").fillna(0)
    merged[f"delta_{candidate_label}_minus_{reference_label}"] = (
        merged[f"count_{candidate_label}"] - merged[f"count_{reference_label}"]
    )
    merged = merged.sort_values(f"delta_{candidate_label}_minus_{reference_label}", ascending=True).reset_index(drop=True)
    return merged


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _style_horizontal_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def plot_overall_metric_comparison(metric_table: pd.DataFrame, labels: list[str], color_map: dict[str, str], output_dir: Path) -> None:
    plt = _get_plt()
    overall = metric_table[metric_table["split"] == "Overall"].copy()
    overall = overall.set_index("experiment").loc[labels].reset_index()

    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    fig.suptitle("Overall Recognition Metrics", fontsize=9, fontweight="bold", y=0.985)

    x_positions = list(range(len(QUALITY_METRICS)))
    width = min(0.68 / max(len(labels), 1), 0.22)
    center_offset = (len(labels) - 1) / 2
    for idx, label in enumerate(labels):
        offsets = [x + (idx - center_offset) * width for x in x_positions]
        values = [float(overall.loc[overall["experiment"] == label, metric].iloc[0]) for metric, _, _ in QUALITY_METRICS]
        ax.bar(offsets, values, width=width, color=color_map[label], alpha=0.9)
        for x, y in zip(offsets, values):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=6.3)

    ax.set_xticks(x_positions, [metric_label for _, metric_label, _ in QUALITY_METRICS])
    ax.set_xlim(-0.42, len(QUALITY_METRICS) - 0.58)
    ax.set_ylim(0.0, 1.13)
    ax.set_ylabel("Score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    _style_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label], alpha=0.9) for label in labels]
    legend = fig.legend(
        handles,
        [display_label(label) for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        frameon=True,
        ncol=len(labels),
        fontsize=6.6,
        fancybox=False,
        borderpad=0.2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.55,
    )
    legend.get_frame().set_edgecolor("#7a8793")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.92)

    fig.tight_layout(rect=(0, 0, 1, 0.83), pad=0.35)
    _save_figure(fig, output_dir, "overall_metric_comparison")
    plt.close(fig)


def plot_latency_comparison(metric_table: pd.DataFrame, labels: list[str], color_map: dict[str, str], output_dir: Path) -> None:
    plt = _get_plt()
    split_order = [str(item) for item in metric_table["split"].cat.categories]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), width_ratios=[1.0, 2.0])
    fig.suptitle("Latency Comparison", fontweight="bold", y=0.98)

    overall = metric_table[metric_table["split"] == "Overall"].copy()
    overall = overall.set_index("experiment").loc[labels].reset_index()
    latency_values = [float(overall.loc[overall["experiment"] == label, "avg_latency_s"].iloc[0]) for label in labels]
    bars = axes[0].bar(
        [display_label(label) for label in labels],
        latency_values,
        color=[color_map[label] for label in labels],
        alpha=0.9,
        width=0.6,
    )
    for bar, value in zip(bars, latency_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}s", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Overall Average Latency")
    axes[0].set_ylabel("Seconds / sample")
    _style_axes(axes[0])

    width = min(0.8 / max(len(labels), 1), 0.25)
    center_offset = (len(labels) - 1) / 2
    x_positions = list(range(len(split_order)))
    for idx, label in enumerate(labels):
        experiment_rows = (
            metric_table[metric_table["experiment"] == label]
            .set_index("split")
            .reindex(split_order)
            .reset_index()
        )
        offsets = [x + (idx - center_offset) * width for x in x_positions]
        values = experiment_rows["avg_latency_s"].astype(float).tolist()
        axes[1].bar(offsets, values, width=width, color=color_map[label], alpha=0.9)
        for x, y in zip(offsets, values):
            axes[1].text(x, y + 0.03, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x_positions, split_order)
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Seconds / sample")
    axes[1].set_title("Per-Split Latency")
    _style_axes(axes[1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label], alpha=0.9) for label in labels]
    legend = fig.legend(
        handles,
        [display_label(label) for label in labels],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.955),
        frameon=True,
        ncol=1,
        title="Experiments",
        fancybox=False,
    )
    legend.get_frame().set_edgecolor("#7a8793")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.96)

    fig.tight_layout(rect=(0, 0, 0.98, 0.94))
    _save_figure(fig, output_dir, "latency_comparison")
    plt.close(fig)


def plot_split_metric_comparison(metric_table: pd.DataFrame, labels: list[str], color_map: dict[str, str], output_dir: Path) -> None:
    plt = _get_plt()
    split_order = [str(item) for item in metric_table["split"].cat.categories]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.65))
    fig.suptitle("Per-Split Comparison", fontsize=8.5, fontweight="bold", y=0.985)

    width = min(0.72 / max(len(labels), 1), 0.22)
    x_positions = list(range(len(split_order)))
    center_offset = (len(labels) - 1) / 2
    for ax, (metric_key, metric_label, bounded_to_one) in zip(axes.flat, QUALITY_METRICS):
        for idx, label in enumerate(labels):
            experiment_rows = (
                metric_table[metric_table["experiment"] == label]
                .set_index("split")
                .reindex(split_order)
                .reset_index()
            )
            offsets = [x + (idx - center_offset) * width for x in x_positions]
            values = experiment_rows[metric_key].astype(float).tolist()
            ax.bar(
                offsets,
                values,
                width=width,
                color=color_map[label],
                alpha=0.9,
            )
            for x, y in zip(offsets, values):
                ax.text(
                    x,
                    y + (0.012 if bounded_to_one else 0.006),
                    f"{y:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.1,
                )

        ax.set_title(metric_label, fontsize=7.1, pad=3)
        ax.set_xticks(x_positions, split_order)
        ax.tick_params(axis="both", labelsize=5.6)
        ax.set_ylim(0.0, 1.05 if bounded_to_one else max(0.3, float(metric_table[metric_key].max()) * 1.12))
        _style_axes(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label], alpha=0.9) for label in labels]
    legend = fig.legend(
        handles,
        [display_label(label) for label in labels],
        loc="upper right",
        bbox_to_anchor=(0.986, 0.955),
        frameon=True,
        ncol=1,
        title="Experiments",
        fontsize=5.5,
        title_fontsize=6.0,
        fancybox=False,
        borderpad=0.3,
        labelspacing=0.35,
        handlelength=1.4,
    )
    legend.get_frame().set_edgecolor("#7a8793")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.96)

    fig.tight_layout(rect=(0, 0, 0.97, 0.935), h_pad=1.0, w_pad=1.0)
    _save_figure(fig, output_dir, "per_split_metric_comparison")
    plt.close(fig)


def plot_outcome_transition(outcome_table: pd.DataFrame, reference_label: str, candidate_label: str, output_dir: Path) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(outcome_table)))
    bottom = [0.0] * len(outcome_table)

    for outcome in OUTCOME_ORDER:
        values = outcome_table[f"{outcome}_rate"].astype(float).tolist()
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=OUTCOME_COLORS[outcome],
            alpha=0.95,
            width=0.65,
            label=OUTCOME_LABELS[outcome],
        )
        bottom = [base + value for base, value in zip(bottom, values)]

    ax.set_xticks(x_positions, outcome_table["split"].astype(str).tolist())
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of Samples")
    ax.set_xlabel("Split")
    ax.set_title(
        f"Outcome Transition: {display_label(reference_label)} -> {display_label(candidate_label)}",
        fontweight="bold",
    )
    _style_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    for x, (_, row) in zip(x_positions, outcome_table.iterrows()):
        fixed = int(row["fixed_by_candidate"])
        regressed = int(row["regressed_after_candidate"])
        ax.text(
            x,
            0.985,
            f"+{fixed} / -{regressed}",
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
        )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save_figure(fig, output_dir, "outcome_transition_comparison")
    plt.close(fig)


def plot_cer_improvement_distribution(pairwise_df: pd.DataFrame, candidate_label: str, output_dir: Path) -> None:
    plt = _get_plt()
    split_order = sorted(dict.fromkeys(pairwise_df["eval_split"].astype(str).tolist()))
    data = [pairwise_df.loc[pairwise_df["eval_split"] == split, "cer_improvement"].tolist() for split in split_order]
    data.append(pairwise_df["cer_improvement"].tolist())
    labels = split_order + ["Overall"]

    fig, ax = plt.subplots(figsize=(11.5, 6))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#28323c", "linewidth": 1.5},
    )
    colors = ["#90be6d", "#43aa8b", "#4d908e", "#577590"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.axhline(0.0, color="#bc4749", linestyle="--", linewidth=1.2)
    ax.set_title(f"CER Improvement Distribution of {display_label(candidate_label)}", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("CER(reference) - CER(candidate)")
    _style_axes(ax)

    fig.tight_layout()
    _save_figure(fig, output_dir, "cer_improvement_distribution")
    plt.close(fig)


def plot_error_bucket_shift(bucket_shift_table: pd.DataFrame, reference_label: str, candidate_label: str, output_dir: Path) -> None:
    plt = _get_plt()
    delta_column = f"delta_{candidate_label}_minus_{reference_label}"
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = []
    for _, row in bucket_shift_table.iterrows():
        value = float(row[delta_column])
        is_positive_shift = value > 0
        is_exact_match_bucket = row["error_bucket"] == "exact_match"
        is_good_shift = (is_exact_match_bucket and is_positive_shift) or ((not is_exact_match_bucket) and value < 0)
        colors.append("#43aa8b" if is_good_shift else "#bc4749")
    ax.barh(bucket_shift_table["error_bucket"], bucket_shift_table[delta_column], color=colors, alpha=0.9)
    ax.axvline(0.0, color="#3f4e5a", linewidth=1.1)
    ax.set_title(
        f"Error Bucket Count Shift: {display_label(candidate_label)} - {display_label(reference_label)}",
        fontweight="bold",
    )
    ax.set_xlabel("Count Delta")
    _style_horizontal_axes(ax)

    for y, value in zip(bucket_shift_table["error_bucket"], bucket_shift_table[delta_column]):
        offset = 6 if value >= 0 else -6
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, y, f"{int(value):+d}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    _save_figure(fig, output_dir, "error_bucket_shift")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    parsed_experiments = parse_experiment_args(args.experiments)
    experiments = [load_experiment_results(label, path) for label, path in parsed_experiments]
    labels = [item[0] for item in parsed_experiments]
    reference_label = args.reference_label or labels[0]
    if reference_label not in labels:
        raise ValueError(f"Unknown --reference-label: {reference_label}. Available labels: {labels}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "experiment_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_table = build_metric_table(experiments)
    color_map = build_color_map(labels)
    metric_table.to_csv(output_dir / "comparison_metrics.csv", index=False)
    plot_overall_metric_comparison(metric_table, labels, color_map, output_dir)
    plot_split_metric_comparison(metric_table, labels, color_map, output_dir)
    plot_latency_comparison(metric_table, labels, color_map, output_dir)

    experiment_by_label = {experiment["label"]: experiment for experiment in experiments}
    reference_experiment = experiment_by_label[reference_label]
    pairwise_index: list[dict[str, object]] = []

    for candidate_label in labels:
        if candidate_label == reference_label:
            continue
        candidate_experiment = experiment_by_label[candidate_label]
        pairwise_output_dir = output_dir / f"vs_{candidate_label}"
        pairwise_output_dir.mkdir(parents=True, exist_ok=True)

        pairwise_df = build_pairwise_comparison(
            reference_experiment["evaluated_all"],
            candidate_experiment["evaluated_all"],
        )
        outcome_table = build_outcome_table(pairwise_df)
        bucket_shift_table = build_bucket_shift_table(
            reference_experiment["bucket_metrics"],
            candidate_experiment["bucket_metrics"],
            reference_label,
            candidate_label,
        )
        pairwise_summary = build_pairwise_summary(pairwise_df, reference_label, candidate_label)

        outcome_table.to_csv(pairwise_output_dir / "outcome_transition_summary.csv", index=False)
        bucket_shift_table.to_csv(pairwise_output_dir / "error_bucket_shift.csv", index=False)
        with open(pairwise_output_dir / "pairwise_summary.json", "w", encoding="utf-8") as handle:
            json.dump(pairwise_summary, handle, indent=2)

        plot_outcome_transition(outcome_table, reference_label, candidate_label, pairwise_output_dir)
        plot_cer_improvement_distribution(pairwise_df, candidate_label, pairwise_output_dir)
        plot_error_bucket_shift(bucket_shift_table, reference_label, candidate_label, pairwise_output_dir)
        pairwise_index.append(
            {
                "reference_experiment": reference_label,
                "candidate_experiment": candidate_label,
                "output_dir": str(pairwise_output_dir),
                **pairwise_summary,
            }
        )

    with open(output_dir / "pairwise_index.json", "w", encoding="utf-8") as handle:
        json.dump(pairwise_index, handle, indent=2)

    print(f"Saved comparison figures to: {output_dir}")


if __name__ == "__main__":
    main()
