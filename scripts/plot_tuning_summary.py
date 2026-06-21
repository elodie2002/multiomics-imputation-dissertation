"""
Plot tuning summaries for DAE, VAE, and GAIN.

This handles CSVs with or without headers.

Usage:
    python plot_tuning_summary.py \
        --dae_csv DAE_tuning_results.csv \
        --vae_csv VAE_tuning_results.csv \
        --gain_csv GAIN_tuning_results.csv \
        --output_dir tuning_summary_plots
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["rmse", "mae", "pearson", "spearman", "runtime_seconds"]

MODEL_COLUMNS = {
    "DAE": [
        "experiment_type", "model", "dataset", "omics", "data_name",
        "miss_rate", "seed", "tuning_param", "tuning_value",
        "hidden_dim", "lr", "weight_decay", "batch_size",
        "epochs", "noise_rate", "dropout",
        "rmse", "mae", "pearson", "spearman", "runtime_seconds"
    ],
    "VAE": [
        "experiment_type", "model", "dataset", "omics", "data_name",
        "miss_rate", "seed", "tuning_param", "tuning_value",
        "hidden_dim", "latent_dim", "lr", "weight_decay",
        "batch_size", "epochs", "noise_rate", "beta", "dropout",
        "rmse", "mae", "pearson", "spearman", "runtime_seconds"
    ],
    "GAIN": [
        "experiment_type", "model", "dataset", "omics", "data_name",
        "miss_rate", "seed", "tuning_param", "tuning_value",
        "batch_size", "hint_rate", "alpha", "iterations",
        "rmse", "mae", "pearson", "spearman", "runtime_seconds"
    ],
}


def looks_like_header(first_row) -> bool:
    """Return True only if first row clearly contains column names."""
    values = {str(x).strip().lower() for x in first_row.tolist()}
    header_keywords = {"rmse", "mae", "pearson", "spearman", "tuning_param", "tuning_value", "data_name"}
    return len(values.intersection(header_keywords)) >= 3


def read_tuning_csv(path: Path, model: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)

    expected_cols = MODEL_COLUMNS[model]

    if looks_like_header(raw.iloc[0]):
        header = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
        df = raw.iloc[1:].copy()
        df.columns = header
    else:
        if raw.shape[1] != len(expected_cols):
            raise ValueError(
                f"{model}: expected {len(expected_cols)} columns, "
                f"but found {raw.shape[1]} in {path}."
            )
        df = raw.copy()
        df.columns = expected_cols

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "missingness": "miss_rate",
        "runtime": "runtime_seconds",
        "runtime_sec": "runtime_seconds",
        "runtime_secs": "runtime_seconds",
        "dataset_name": "data_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Force model name to be consistent.
    df["model"] = model

    numeric_cols = [
        "miss_rate", "seed", "tuning_value",
        "hidden_dim", "latent_dim", "lr", "weight_decay",
        "batch_size", "epochs", "noise_rate", "beta", "dropout",
        "hint_rate", "alpha", "iterations",
        "rmse", "mae", "pearson", "spearman", "runtime_seconds"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"model", "data_name", "miss_rate", "seed", "tuning_param", "tuning_value", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{model}: missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [m for m in METRICS if m in df.columns]
    summary = (
        df.groupby(["model", "tuning_param", "tuning_value"], dropna=False)[available_metrics]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if x != ""]).strip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def label_value(x) -> str:
    if pd.isna(x):
        return "NA"
    try:
        x = float(x)
        if abs(x) >= 100:
            return str(int(x)) if x.is_integer() else f"{x:g}"
        return f"{x:g}"
    except Exception:
        return str(x)


def selected_values_for_model(model: str) -> Dict[str, float]:
    # Selected final settings. 
    if model == "DAE":
        return {
            "hidden_dim": 256,
            "epochs": 100,
            "noise_rate": 0.1,
            "dropout": 0.2,
        }
    if model == "VAE":
        return {
            "latent_dim": 32,
            "beta": 1e-6,
            "epochs": 100,
            "dropout": 0.1,
        }
    if model == "GAIN":
        return {
            "iterations": 500,
            "alpha": 100,
            "hint_rate": 0.9,
        }
    return {}


def save_sensitivity_figure(
    df: pd.DataFrame,
    model: str,
    metric: str,
    output_dir: Path,
    selected: Optional[Dict[str, float]] = None,
) -> None:
    selected = selected or {}
    sub_model = df[df["model"] == model].copy()
    if metric not in sub_model.columns:
        return

    tuning_params = sorted(sub_model["tuning_param"].dropna().astype(str).unique())
    if not tuning_params:
        return

    n = len(tuning_params)
    fig_width = max(4.5 * n, 7)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4.6), squeeze=False)
    axes = axes[0]

    for ax, param in zip(axes, tuning_params):
        sub = sub_model[sub_model["tuning_param"].astype(str) == param].copy()
        sub = sub.dropna(subset=[metric, "tuning_value"])
        if sub.empty:
            ax.set_visible(False)
            continue

        grouped = (
            sub.groupby("tuning_value")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("tuning_value")
            .reset_index(drop=True)
        )

        x = list(range(len(grouped)))
        y = grouped["mean"]
        yerr = grouped["std"].fillna(0)

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([label_value(v) for v in grouped["tuning_value"]], rotation=30, ha="right")
        ax.set_title(param)
        ax.set_xlabel("Tested value")
        ax.set_ylabel(metric.upper() if metric != "runtime_seconds" else "Runtime (seconds)")
        ax.grid(True, axis="y", alpha=0.3)

        if param in selected:
            selected_value = float(selected[param])
            matched_positions = [
                i for i, v in enumerate(grouped["tuning_value"])
                if abs(float(v) - selected_value) < 1e-12
            ]
            if matched_positions:
                i = matched_positions[0]
                ax.scatter(
                    i, grouped.loc[i, "mean"],
                    s=130, marker="o", facecolors="none",
                    edgecolors="black", linewidths=2.0, zorder=5
                )

    direction_note = ""
    if metric in ["rmse", "mae"]:
        direction_note = "lower is better"
    elif metric == "runtime_seconds":
        direction_note = "lower is faster"
    elif metric in ["pearson", "spearman"]:
        direction_note = "higher is better"

    fig.suptitle(f"{model} Hyperparameter Sensitivity ({metric.upper()})", fontsize=16)
    fig.text(
        0.5, 0.01,
        f"Dot = mean; error bar = standard deviation; black circle = selected final setting; {direction_note}.",
        ha="center", fontsize=10
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(output_dir / f"{model}_tuning_{metric}_summary.png", dpi=300)
    plt.close(fig)


def print_completeness(df: pd.DataFrame) -> None:
    print("\n=== Completeness check ===")
    for model, sub in df.groupby("model"):
        print(f"\n{model}")
        print("Rows:", len(sub))
        print("Tuning params:", sorted(sub["tuning_param"].astype(str).unique()))
        print("Cases:", sorted(sub["data_name"].astype(str).unique()))
        print("Missingness:", sorted(sub["miss_rate"].dropna().unique()))
        print("Seeds:", sorted(sub["seed"].dropna().unique()))
        print("Rows by parameter:")
        print(sub.groupby("tuning_param").size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dae_csv", required=True)
    parser.add_argument("--vae_csv", required=True)
    parser.add_argument("--gain_csv", required=True)
    parser.add_argument("--output_dir", default="tuning_summary_plots")
    parser.add_argument("--make_appendix", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.concat(
        [
            read_tuning_csv(Path(args.dae_csv), "DAE"),
            read_tuning_csv(Path(args.vae_csv), "VAE"),
            read_tuning_csv(Path(args.gain_csv), "GAIN"),
        ],
        ignore_index=True,
    )

    print_completeness(df_all)

    summary_path = output_dir / "tuning_summary_all_models.csv"
    summarize(df_all).to_csv(summary_path, index=False)

    for model in ["DAE", "VAE", "GAIN"]:
        save_sensitivity_figure(
            df_all, model=model, metric="rmse", output_dir=output_dir,
            selected=selected_values_for_model(model)
        )

    if args.make_appendix:
        for metric in ["mae", "pearson", "spearman", "runtime_seconds"]:
            for model in ["DAE", "VAE", "GAIN"]:
                save_sensitivity_figure(
                    df_all, model=model, metric=metric, output_dir=output_dir,
                    selected=selected_values_for_model(model)
                )

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")


if __name__ == "__main__":
    main()