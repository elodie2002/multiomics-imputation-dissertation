from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DAE_PATH = Path("DAE_final_results.csv")
VAE_PATH = Path("VAE_final_results.csv")
GAIN_PATH = Path("GAIN_final_results.csv")

OUTPUT_DIR = Path("final_analysis_outputs/sensitivity_checks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_ORDER = ["DAE", "VAE", "GAIN"]

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


def load_dae(path: Path) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=1, header=None)
    first_cell = str(preview.iloc[0, 0]).strip().lower()

    if first_cell == "experiment_type":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(DAE_COLUMNS):
            raise ValueError(
                f"{path} has {df.shape[1]} columns, expected {len(DAE_COLUMNS)}."
            )
        df.columns = DAE_COLUMNS

    df["model"] = "DAE"
    return df


def load_standard(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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

    df["dataset"] = df["dataset"].astype(str)
    df["omics"] = df["omics"].astype(str)
    df["model"] = df["model"].astype(str)

    return df


dae = clean_dataframe(load_dae(DAE_PATH))
vae = clean_dataframe(load_standard(VAE_PATH, "VAE"))
gain = clean_dataframe(load_standard(GAIN_PATH, "GAIN"))

df = pd.concat([dae, vae, gain], ignore_index=True)

# Remove the severe-risk settings:
# BRCA/COAD miRNA at 0.5 and 0.7 missingness.
exclude = (
    (df["omics"] == "miRNA") &
    (df["dataset"].isin(["BRCA", "COAD"])) &
    (df["miss_rate"].isin([0.5, 0.7]))
)

filtered = df[~exclude].copy()

print("Original rows:", len(df))
print("Excluded rows:", int(exclude.sum()))
print("Remaining rows:", len(filtered))
print("\nRows by model after filtering:")
print(filtered.groupby("model").size())

print("\nRMSE summary after excluding severe-risk settings:")
summary = filtered.groupby("model")["rmse"].agg(["mean", "median", "std", "count"]).reindex(MODELS_ORDER)
print(summary)

# Save summary table.
summary.to_csv(OUTPUT_DIR / "rmse_summary_excluding_brca_coad_mirna_high_missingness.csv")

# Plot boxplot.
data = [
    filtered[filtered["model"] == model]["rmse"].dropna().values
    for model in MODELS_ORDER
]

plt.figure(figsize=(7.4, 5.2))
plt.boxplot(data, tick_labels=MODELS_ORDER)
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE Distribution Excluding BRCA/COAD miRNA High-Missingness Settings")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

out_path = OUTPUT_DIR / "rmse_boxplot_excluding_brca_coad_mirna_high_missingness.png"
plt.savefig(out_path, dpi=300)
plt.close()

print("\nSaved plot to:", out_path)