
"""
Deep analysis for final imputation benchmark results.

Purpose:
    This script identifies:
    1. Worst-performing settings for each model
    2. Best-performing settings for each model
    3. Worst GAIN failure cases
    4. Dataset × omics interaction heatmaps
    5. Runtime drivers by dataset and omics type

Inputs expected in the same folder:
    DAE_final_results.csv
    VAE_final_results.csv
    GAIN_final_results.csv

Run:
    python plot_further_analysis.py

Outputs:
    final_deep_insight_outputs/
        tables/
        figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DAE_PATH = Path("DAE_final_results.csv")
VAE_PATH = Path("VAE_final_results.csv")
GAIN_PATH = Path("GAIN_final_results.csv")

OUTPUT_DIR = Path("final_deep_insight_outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, TABLE_DIR, FIG_DIR]:
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

MODELS_ORDER = ["DAE", "VAE", "GAIN"]
DATASET_ORDER = ["BRCA", "COAD", "GBM", "LGG", "OV"]
OMICS_ORDER = ["CNV", "Methy", "miRNA", "mRNA"]
MISSINGNESS_ORDER = [0.3, 0.5, 0.7]
METRICS = ["rmse", "mae", "pearson", "spearman", "runtime_seconds"]


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
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    df = df[COMMON_COLUMNS].copy()

    for col in ["miss_rate", "seed", "rmse", "mae", "pearson", "spearman", "runtime_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model"] = pd.Categorical(df["model"], categories=MODELS_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["omics"] = pd.Categorical(df["omics"], categories=OMICS_ORDER, ordered=True)
    df["miss_rate"] = pd.Categorical(df["miss_rate"], categories=MISSINGNESS_ORDER, ordered=True)

    return df


def load_all_results() -> pd.DataFrame:
    dae = load_dae(DAE_PATH)
    vae = load_standard(VAE_PATH, "VAE")
    gain = load_standard(GAIN_PATH, "GAIN")
    return pd.concat([clean_dataframe(dae), clean_dataframe(vae), clean_dataframe(gain)], ignore_index=True)


def save_extreme_case_tables(df: pd.DataFrame, top_n: int = 10) -> None:
    cols = ["model", "dataset", "omics", "data_name", "miss_rate", "seed", "rmse", "mae", "pearson", "spearman", "runtime_seconds"]

    df.sort_values("rmse", ascending=False)[cols].head(top_n).to_csv(
        TABLE_DIR / "top_worst_rmse_cases_overall.csv", index=False
    )
    df.sort_values("rmse", ascending=True)[cols].head(top_n).to_csv(
        TABLE_DIR / "top_best_rmse_cases_overall.csv", index=False
    )
    df.sort_values("spearman", ascending=True)[cols].head(top_n).to_csv(
        TABLE_DIR / "top_worst_spearman_cases_overall.csv", index=False
    )

    gain = df[df["model"] == "GAIN"].copy()
    gain.sort_values("rmse", ascending=False)[cols].head(top_n).to_csv(
        TABLE_DIR / "top_worst_gain_rmse_cases.csv", index=False
    )
    gain.sort_values("spearman", ascending=True)[cols].head(top_n).to_csv(
        TABLE_DIR / "top_worst_gain_spearman_cases.csv", index=False
    )

    (
        df.sort_values("rmse", ascending=True)
        .groupby("model", observed=False)
        .head(5)[cols]
        .to_csv(TABLE_DIR / "top_5_best_rmse_cases_by_model.csv", index=False)
    )


def save_group_summary_tables(df: pd.DataFrame) -> None:
    groupings = {
        "dataset_omics": ["model", "dataset", "omics"],
        "dataset_omics_missingness": ["model", "dataset", "omics", "miss_rate"],
        "omics_missingness": ["model", "omics", "miss_rate"],
        "dataset_missingness": ["model", "dataset", "miss_rate"],
    }

    for name, cols in groupings.items():
        summary = (
            df.groupby(cols, observed=False)[METRICS]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )
        summary.to_csv(TABLE_DIR / f"summary_by_{name}.csv", index=False)

    (
        df.groupby(["model", "dataset", "omics", "miss_rate"], observed=False)[["rmse", "mae", "pearson", "spearman"]]
        .mean()
        .reset_index()
        .sort_values("rmse", ascending=False)
        .head(20)
        .to_csv(TABLE_DIR / "top_20_hardest_mean_rmse_settings.csv", index=False)
    )

    (
        df.groupby(["model", "dataset", "omics", "miss_rate"], observed=False)[["rmse", "mae", "pearson", "spearman"]]
        .mean()
        .reset_index()
        .sort_values("rmse", ascending=True)
        .head(20)
        .to_csv(TABLE_DIR / "top_20_easiest_mean_rmse_settings.csv", index=False)
    )


def save_heatmap(matrix: pd.DataFrame, title: str, filename: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(matrix.values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)


def save_dataset_omics_heatmaps(df: pd.DataFrame) -> None:
    for model in MODELS_ORDER:
        sub = df[df["model"] == model].copy()

        for metric, label in [("rmse", "Mean RMSE"), ("spearman", "Mean Spearman")]:
            matrix = (
                sub.groupby(["dataset", "omics"], observed=False)[metric]
                .mean()
                .reset_index()
                .pivot(index="dataset", columns="omics", values=metric)
            )
            matrix = matrix.reindex(index=DATASET_ORDER, columns=OMICS_ORDER)
            save_heatmap(
                matrix,
                f"{model}: Dataset × Omics Interaction ({label})",
                f"heatmap_{model.lower()}_dataset_omics_{metric}.png",
                label,
            )


def save_gain_failure_counts(df: pd.DataFrame) -> None:
    gain = df[df["model"] == "GAIN"].copy()
    threshold = gain["rmse"].quantile(0.90)
    failures = gain[gain["rmse"] >= threshold].copy()

    failures.to_csv(TABLE_DIR / "gain_severe_failure_cases_top10_percent_rmse.csv", index=False)

    count_table = failures.groupby(["omics", "miss_rate"], observed=False).size().reset_index(name="failure_count")
    pivot = count_table.pivot(index="omics", columns="miss_rate", values="failure_count")
    pivot = pivot.reindex(index=OMICS_ORDER, columns=MISSINGNESS_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of severe GAIN failure cases")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=9)

    ax.set_xlabel("Missing Rate")
    ax.set_ylabel("Omics Type")
    ax.set_title("Location of Severe GAIN Failure Cases (Top 10% RMSE)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gain_failure_counts_by_omics_missingness.png", dpi=300)
    plt.close(fig)


def save_runtime_driver_figures(df: pd.DataFrame) -> None:
    for group_col, order in [("omics", OMICS_ORDER), ("dataset", DATASET_ORDER)]:
        summary = df.groupby(["model", group_col], observed=False)["runtime_seconds"].agg(["mean", "std"]).reset_index()
        mean_pivot = summary.pivot(index=group_col, columns="model", values="mean")
        std_pivot = summary.pivot(index=group_col, columns="model", values="std")

        mean_pivot = mean_pivot.reindex(order)[MODELS_ORDER]
        std_pivot = std_pivot.reindex(order)[MODELS_ORDER]

        ax = mean_pivot.plot(kind="bar", yerr=std_pivot, capsize=3, figsize=(8.5, 5.2))
        ax.set_xlabel(group_col.capitalize())
        ax.set_ylabel("Runtime per run (seconds)")
        ax.set_title(f"Runtime by {group_col.capitalize()}")
        ax.legend(title="Model")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"runtime_by_{group_col}.png", dpi=300)
        plt.close()


def save_best_worst_bar(df: pd.DataFrame) -> None:
    setting_summary = (
        df.groupby(["model", "dataset", "omics", "miss_rate"], observed=False)["rmse"]
        .mean()
        .reset_index()
    )

    hardest = setting_summary.sort_values("rmse", ascending=False).head(10).copy()
    easiest = setting_summary.sort_values("rmse", ascending=True).head(10).copy()

    for name, sub, title in [
        ("hardest", hardest, "Top 10 Hardest Mean RMSE Settings"),
        ("easiest", easiest, "Top 10 Easiest Mean RMSE Settings"),
    ]:
        labels = [f"{r.model}\n{r.dataset}-{r.omics}\nmiss={r.miss_rate}" for r in sub.itertuples()]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(sub))
        ax.bar(x, sub["rmse"].values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean RMSE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}_mean_rmse_settings.png", dpi=300)
        plt.close(fig)


def main() -> None:
    df = load_all_results()
    print("Loaded rows:", len(df))
    print(df.groupby("model", observed=False).size())

    df.to_csv(OUTPUT_DIR / "combined_final_results_for_deep_insight.csv", index=False)

    save_extreme_case_tables(df, top_n=10)
    save_group_summary_tables(df)
    save_dataset_omics_heatmaps(df)
    save_gain_failure_counts(df)
    save_runtime_driver_figures(df)
    save_best_worst_bar(df)

    print("\nDone.")
    print(f"Tables saved to: {TABLE_DIR.resolve()}")
    print(f"Figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
