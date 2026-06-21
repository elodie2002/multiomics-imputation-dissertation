import argparse
import csv
import os
import time
import numpy as np

from data_loader import data_loader
from vae import train_and_impute
from metrics import rmse_loss, mae_loss, pearson_masked, spearman_masked


CSV_COLUMNS = [
    "experiment_type", "model", "dataset", "omics", "data_name",
    "miss_rate", "seed", "tuning_param", "tuning_value",
    "hidden_dim", "latent_dim", "lr", "weight_decay", "batch_size",
    "epochs", "noise_rate", "beta", "dropout",
    "rmse", "mae", "pearson", "spearman", "runtime_seconds",
]


def split_data_name(data_name):
    parts = data_name.split("_", 1)
    dataset = parts[0]
    omics = parts[1] if len(parts) > 1 else ""
    return dataset, omics


def append_csv_row(output_csv, row):
    if output_csv is None:
        return

    file_exists = os.path.exists(output_csv)
    has_content = file_exists and os.path.getsize(output_csv) > 0

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not has_content:
            writer.writeheader()
        writer.writerow(row)


def run_one_seed(args, seed):
    data_name = args.data_name
    miss_rate = args.miss_rate

    # New reproducible mask per seed
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, seed=seed)

    start_time = time.time()
    imputed_data_x = train_and_impute(
        miss_data_x,
        data_m,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        noise_rate=args.noise_rate,
        beta=args.beta,
        dropout=args.dropout,
        seed=seed,
        device=args.device,
    )
    runtime_seconds = time.time() - start_time

    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    mae = mae_loss(ori_data_x, imputed_data_x, data_m)
    pearson = pearson_masked(ori_data_x, imputed_data_x, data_m)
    spearman = spearman_masked(ori_data_x, imputed_data_x, data_m)

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "runtime_seconds": runtime_seconds,
    }


def main(args):
    data_name = args.data_name
    miss_rate = args.miss_rate
    seeds = args.seeds
    dataset, omics = split_data_name(data_name)

    print("\n")
    print(data_name)
    print(f"Missingness: {miss_rate}")
    print(f"Seeds: {seeds}")
    print(f"Tuning: {args.tuning_param} = {args.tuning_value}")
    print("\n")

    all_results = []

    for seed in seeds:
        print(f"Running seed: {seed}")
        metrics = run_one_seed(args, seed)
        all_results.append(metrics)

        print("RMSE Performance: " + str(np.round(metrics["rmse"], 4)))
        print("MAE Performance: " + str(np.round(metrics["mae"], 4)))
        print("Pearson Performance: " + str(np.round(metrics["pearson"], 4)))
        print("Spearman Performance: " + str(np.round(metrics["spearman"], 4)))
        print("Runtime seconds: " + str(np.round(metrics["runtime_seconds"], 2)))

        row = {
            "experiment_type": args.experiment_type,
            "model": "VAE",
            "dataset": dataset,
            "omics": omics,
            "data_name": data_name,
            "miss_rate": miss_rate,
            "seed": seed,
            "tuning_param": args.tuning_param,
            "tuning_value": args.tuning_value,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "noise_rate": args.noise_rate,
            "beta": args.beta,
            "dropout": args.dropout,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "runtime_seconds": metrics["runtime_seconds"],
        }
        append_csv_row(args.output_csv, row)

    print("\n=== Summary across seeds ===")
    for metric_name in ["rmse", "mae", "pearson", "spearman", "runtime_seconds"]:
        values = np.array([r[metric_name] for r in all_results], dtype=float)
        print(f"{metric_name.upper()} Mean: {np.round(np.nanmean(values), 4)}")
        print(f"{metric_name.upper()} Median: {np.round(np.nanmedian(values), 4)}")
        print(f"{metric_name.upper()} Std: {np.round(np.nanstd(values), 4)}")
        print()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="BRCA_mRNA", type=str)
    parser.add_argument("--miss_rate", default=0.5, type=float)

    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--noise_rate", default=0.1, type=float)
    parser.add_argument("--beta", default=1e-6, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output_csv", default=None, type=str)
    parser.add_argument("--experiment_type", default="single", type=str)
    parser.add_argument("--tuning_param", default="none", type=str)
    parser.add_argument("--tuning_value", default="none", type=str)

    args = parser.parse_args()
    main(args)