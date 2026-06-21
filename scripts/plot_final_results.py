"""
final result plotting script for DAE, VAE, and GAIN.

This generates:
- main_figures: figures with SD error bars
- appendix_figures: supplementary figures with SD error bars
- annotated_figures: numeric labels for checking 

Inputs expected in the same folder:
    DAE_final_results.csv
    VAE_final_results.csv
    GAIN_final_results.csv

Run:
    python plot_final_results.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DAE_PATH = Path("DAE_final_results.csv")
VAE_PATH = Path("VAE_final_results.csv")
GAIN_PATH = Path("GAIN_final_results.csv")

OUTPUT_DIR = Path("final_analysis_outputs")
MAIN_FIG_DIR = OUTPUT_DIR / "main_figures"
APPENDIX_FIG_DIR = OUTPUT_DIR / "appendix_figures"
ANNOTATED_FIG_DIR = OUTPUT_DIR / "annotated_figures"

for d in [OUTPUT_DIR, MAIN_FIG_DIR, APPENDIX_FIG_DIR, ANNOTATED_FIG_DIR]:
    d.mkdir(exist_ok=True)

DAE_COLUMNS = [
    "experiment_type", "model", "dataset", "omics", "data_name", "miss_rate", "seed",
    "tuning_param", "tuning_value", "hidden_dim", "lr", "weight_decay", "batch_size",
    "epochs", "noise_rate", "dropout", "rmse", "mae", "pearson", "spearman",
    "runtime_seconds",
]

COMMON_COLUMNS = [
    "experiment_type", "model", "dataset", "omics", "data_name", "miss_rate", "seed",
    "rmse", "mae", "pearson", "spearman", "runtime_seconds",
]

METRICS = ["rmse", "mae", "pearson", "spearman", "runtime_seconds"]
PERFORMANCE_METRICS = ["rmse", "mae", "pearson", "spearman"]
MODELS_ORDER = ["DAE", "VAE", "GAIN"]
MISSINGNESS_ORDER = [0.3, 0.5, 0.7]

METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "pearson": "Pearson Correlation",
    "spearman": "Spearman Correlation",
    "runtime_seconds": "Runtime per run (seconds)",
}


def fmt(metric: str, value: float) -> str:
    if pd.isna(value):
        return "NA"
    if metric == "runtime_seconds":
        return f"{value:.1f}"
    return f"{value:.3f}"


def load_dae(path: Path) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=1, header=None)
    first_cell = str(preview.iloc[0, 0]).strip().lower()
    if first_cell == "experiment_type":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(DAE_COLUMNS):
            raise ValueError(f"{path} has {df.shape[1]} columns, expected {len(DAE_COLUMNS)}.")
        df.columns = DAE_COLUMNS
    df["model"] = "DAE"
    return df


def load_standard(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rmse" not in df.columns:
        raise ValueError(f"{path} does not appear to have a header row.")
    df["model"] = model_name
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "missingness" in df.columns and "miss_rate" not in df.columns:
        df = df.rename(columns={"missingness": "miss_rate"})

    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df[COMMON_COLUMNS].copy()
    for col in ["miss_rate", "seed", "rmse", "mae", "pearson", "spearman", "runtime_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model"] = pd.Categorical(df["model"], categories=MODELS_ORDER, ordered=True)
    df["miss_rate"] = pd.Categorical(df["miss_rate"], categories=MISSINGNESS_ORDER, ordered=True)
    return df


def validate_counts(df: pd.DataFrame) -> None:
    print("\n=== Completeness check ===")
    counts = df.groupby("model", observed=False).size()
    print(counts)
    expected = 5 * 4 * 3 * 5
    for model in MODELS_ORDER:
        actual = int(counts.get(model, 0))
        status = "OK" if actual == expected else "CHECK"
        print(f"{model}: {actual}/{expected} rows [{status}]")
    print("\nRows by model and missingness:")
    print(df.groupby(["model", "miss_rate"], observed=False).size())


def make_summary(df: pd.DataFrame, group_cols: list[str], output_path: Path) -> None:
    summary = (
        df.groupby(group_cols, observed=False)[METRICS]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(output_path, index=False)


def save_readable_mean_std(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["model", "miss_rate"], observed=False)[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    rows = []
    for _, row in grouped.iterrows():
        out = {"model": row["model"], "miss_rate": row["miss_rate"]}
        for metric in METRICS:
            out[f"{metric}_mean±std"] = f"{row[(metric, 'mean')]:.4f} ± {row[(metric, 'std')]:.4f}"
        rows.append(out)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "summary_mean_std_by_missingness_readable.csv", index=False)


def save_line_metric_vs_missingness(df: pd.DataFrame, metric: str, filename: str, fig_dir: Path, annotated: bool = False) -> None:
    summary = (
        df.groupby(["model", "miss_rate"], observed=False)[metric]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    ylabel = METRIC_LABELS[metric]
    suffix = "_annotated" if annotated else ""
    out_name = filename.replace(".png", f"{suffix}.png")
    plt.figure(figsize=(10.5, 6.2) if annotated else (7.8, 5.2))
    y_min = summary["mean"].min()
    y_max = summary["mean"].max()
    y_span = max(y_max - y_min, 1e-6)
    model_offsets = {"DAE": -0.025, "VAE": 0.0, "GAIN": 0.025}

    for idx, model in enumerate(MODELS_ORDER):
        sub = summary[summary["model"] == model].copy()
        sub["x"] = sub["miss_rate"].astype(float)
        plt.errorbar(sub["x"], sub["mean"], yerr=sub["std"], marker="o", capsize=4, label=model, linewidth=2)

        if annotated:
            for _, row in sub.iterrows():
                x = float(row["x"]) + model_offsets.get(model, 0)
                y = float(row["mean"])
                sd = float(row["std"])
                label = f"{model}: {fmt(metric, y)}±{fmt(metric, sd)}"
                if metric in ["pearson", "spearman"]:
                    y_text = y - y_span * (0.025 + idx * 0.008)
                    va = "top"
                else:
                    y_text = y + y_span * (0.025 + idx * 0.008)
                    va = "bottom"
                plt.text(x, y_text, label, fontsize=7, ha="center", va=va)

            plt.text(
                0.01, 0.98, "Dot = mean; error bar = standard deviation",
                transform=plt.gca().transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.12),
            )

    plt.xlabel("Missing Rate")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Missingness")
    plt.xticks(MISSINGNESS_ORDER)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / out_name, dpi=300)
    plt.close()


def save_grouped_bar(df: pd.DataFrame, group_col: str, metric: str, filename: str, fig_dir: Path, annotated: bool = False) -> None:
    summary = df.groupby([group_col, "model"], observed=False)[metric].agg(["mean", "std"]).reset_index()
    mean_pivot = summary.pivot(index=group_col, columns="model", values="mean")
    std_pivot = summary.pivot(index=group_col, columns="model", values="std")

    if group_col == "omics":
        preferred = ["CNV", "Methy", "miRNA", "mRNA"]
        order = [x for x in preferred if x in mean_pivot.index]
        mean_pivot = mean_pivot.loc[order]
        std_pivot = std_pivot.loc[order]
    else:
        mean_pivot = mean_pivot.sort_index()
        std_pivot = std_pivot.loc[mean_pivot.index]

    mean_pivot = mean_pivot[MODELS_ORDER]
    std_pivot = std_pivot[MODELS_ORDER]

    ylabel = f"Mean {METRIC_LABELS[metric]}"
    title = f"{METRIC_LABELS[metric]} by {group_col.capitalize()}"
    suffix = "_annotated" if annotated else ""
    out_name = filename.replace(".png", f"{suffix}.png")

    ax = mean_pivot.plot(
        kind="bar",
        yerr=std_pivot,
        capsize=3,
        figsize=(11, 6) if annotated else (8.5, 5.2),
    )
    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)

    if annotated:
        for container in ax.containers:
            if not hasattr(container, "patches"):
                continue
            for bar in container.patches:
                h = bar.get_height()
                if pd.isna(h):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )
        ax.text(
            0.01, 0.98, "Bar = mean; error bar = standard deviation",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
        )

    plt.tight_layout()
    plt.savefig(fig_dir / out_name, dpi=300)
    plt.close()


def save_boxplot_by_model(df: pd.DataFrame, metric: str, filename: str, fig_dir: Path, annotated: bool = False) -> None:
    data = [df[df["model"] == model][metric].dropna().values for model in MODELS_ORDER]
    ylabel = METRIC_LABELS[metric]
    title = f"{ylabel} Distribution by Model"
    suffix = "_annotated" if annotated else ""
    out_name = filename.replace(".png", f"{suffix}.png")

    plt.figure(figsize=(8.8, 6) if annotated else (7.4, 5.2))
    plt.boxplot(data, tick_labels=MODELS_ORDER)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    if metric in ["pearson", "spearman"]:
        ymin = max(-0.05, min([np.nanmin(x) for x in data if len(x) > 0]) - 0.05)
        plt.ylim(ymin, 1.02)

    if annotated:
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-6)
        for i, model in enumerate(MODELS_ORDER, start=1):
            values = pd.Series(df[df["model"] == model][metric].dropna().values)
            mean = values.mean()
            median = values.median()
            std = values.std()
            label = f"mean={fmt(metric, mean)}\nmedian={fmt(metric, median)}\nsd={fmt(metric, std)}"
            plt.text(
                i + 0.18, median + y_span * 0.03, label,
                fontsize=8, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
            )
        ax.text(
            0.01, 0.98, "Box line = median; box = middle 50%; whiskers = non-outlier range",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
        )

    plt.tight_layout()
    plt.savefig(fig_dir / out_name, dpi=300)
    plt.close()


def save_runtime_bar_by_model(df: pd.DataFrame, fig_dir: Path, annotated: bool = False) -> None:
    summary = df.groupby("model", observed=False)["runtime_seconds"].agg(["mean", "median", "std"]).reindex(MODELS_ORDER)
    filename = "13_runtime_by_model_annotated.png" if annotated else "13_runtime_by_model.png"
    plt.figure(figsize=(8.8, 6) if annotated else (7.4, 5.2))
    x = np.arange(len(MODELS_ORDER))
    plt.bar(x, summary["mean"], yerr=summary["std"], capsize=4)
    plt.xticks(x, MODELS_ORDER)
    plt.xlabel("Model")
    plt.ylabel("Runtime per run (seconds)")
    plt.title("Runtime Comparison by Model")
    plt.grid(axis="y", alpha=0.25)

    if annotated:
        max_mean = max(summary["mean"].max(), 1e-6)
        for i, model in enumerate(MODELS_ORDER):
            mean = summary.loc[model, "mean"]
            median = summary.loc[model, "median"]
            std = summary.loc[model, "std"]
            label = f"mean={mean:.1f}s\nmedian={median:.1f}s\nsd={std:.1f}s"
            plt.text(
                i, mean + std + max_mean * 0.03, label,
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
            )
        plt.gca().text(
            0.01, 0.98, "Bar = mean runtime; error bar = standard deviation",
            transform=plt.gca().transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
        )

    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=300)
    plt.close()


def main() -> None:
    dae = load_dae(DAE_PATH)
    vae = load_standard(VAE_PATH, "VAE")
    gain = load_standard(GAIN_PATH, "GAIN")

    df = pd.concat([clean_dataframe(dae), clean_dataframe(vae), clean_dataframe(gain)], ignore_index=True)

    validate_counts(df)

    df.to_csv(OUTPUT_DIR / "combined_final_results.csv", index=False)
    make_summary(df, ["model"], OUTPUT_DIR / "summary_by_model_overall.csv")
    make_summary(df, ["model", "miss_rate"], OUTPUT_DIR / "summary_by_model_missingness.csv")
    make_summary(df, ["model", "dataset"], OUTPUT_DIR / "summary_by_model_dataset.csv")
    make_summary(df, ["model", "omics"], OUTPUT_DIR / "summary_by_model_omics.csv")
    save_readable_mean_std(df)

    save_boxplot_by_model(df, "rmse", "01_overall_rmse_distribution_by_model.png", MAIN_FIG_DIR)
    save_boxplot_by_model(df, "mae", "02_overall_mae_distribution_by_model.png", MAIN_FIG_DIR)
    save_boxplot_by_model(df, "pearson", "03_overall_pearson_distribution_by_model.png", MAIN_FIG_DIR)
    save_boxplot_by_model(df, "spearman", "04_overall_spearman_distribution_by_model.png", MAIN_FIG_DIR)

    save_line_metric_vs_missingness(df, "rmse", "05_rmse_vs_missingness.png", MAIN_FIG_DIR)
    save_line_metric_vs_missingness(df, "mae", "06_mae_vs_missingness.png", MAIN_FIG_DIR)
    save_line_metric_vs_missingness(df, "pearson", "07_pearson_vs_missingness.png", MAIN_FIG_DIR)
    save_line_metric_vs_missingness(df, "spearman", "08_spearman_vs_missingness.png", MAIN_FIG_DIR)

    save_grouped_bar(df, "omics", "rmse", "09_rmse_by_omics.png", MAIN_FIG_DIR)
    save_grouped_bar(df, "omics", "spearman", "10_spearman_by_omics.png", MAIN_FIG_DIR)
    save_grouped_bar(df, "dataset", "rmse", "11_rmse_by_dataset.png", MAIN_FIG_DIR)
    save_grouped_bar(df, "dataset", "spearman", "12_spearman_by_dataset.png", MAIN_FIG_DIR)

    save_runtime_bar_by_model(df, MAIN_FIG_DIR)
    save_boxplot_by_model(df, "runtime_seconds", "14_runtime_distribution_by_model.png", MAIN_FIG_DIR)

    for group_col in ["omics", "dataset"]:
        for metric in ["mae", "pearson"]:
            save_grouped_bar(df, group_col, metric, f"appendix_{metric}_by_{group_col}.png", APPENDIX_FIG_DIR)

    save_line_metric_vs_missingness(df, "runtime_seconds", "appendix_runtime_vs_missingness.png", APPENDIX_FIG_DIR)

    # Annotated versions for checking exact values only.
    for metric in PERFORMANCE_METRICS:
        save_boxplot_by_model(df, metric, f"annotated_overall_{metric}_distribution_by_model.png", ANNOTATED_FIG_DIR, annotated=True)
        save_line_metric_vs_missingness(df, metric, f"annotated_{metric}_vs_missingness.png", ANNOTATED_FIG_DIR, annotated=True)

    for group_col in ["omics", "dataset"]:
        for metric in ["rmse", "spearman"]:
            save_grouped_bar(df, group_col, metric, f"annotated_{metric}_by_{group_col}.png", ANNOTATED_FIG_DIR, annotated=True)

    save_runtime_bar_by_model(df, ANNOTATED_FIG_DIR, annotated=True)
    save_boxplot_by_model(df, "runtime_seconds", "annotated_runtime_distribution_by_model.png", ANNOTATED_FIG_DIR, annotated=True)

    print("\nDone.")
    print(f"Combined results and summaries saved to: {OUTPUT_DIR.resolve()}")
    print(f"Main figures saved to: {MAIN_FIG_DIR.resolve()}")
    print(f"Appendix figures saved to: {APPENDIX_FIG_DIR.resolve()}")
    print(f"Annotated checking figures saved to: {ANNOTATED_FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
