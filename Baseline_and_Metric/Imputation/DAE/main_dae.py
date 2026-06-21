import argparse
import csv
import os
import time
import numpy as np

from data_loader import data_loader
from dae import train_and_impute
from metrics import rmse_loss, mae_loss, pearson_masked, spearman_masked


RESULT_FIELDS = [
    "experiment_type",
    "model",
    "dataset",
    "omics",
    "data_name",
    "missingness",
    "seed",
    "tuning_param",
    "tuning_value",
    "hidden_dim",
    "lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "noise_rate",
    "dropout",
    "rmse",
    "mae",
    "pearson",
    "spearman",
    "runtime_seconds",
]


def append_csv(path, row):
    """Append one result row to CSV, creating the header if needed."""
    if path is None:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_metric(name, values):
    values = np.array(values, dtype=float)
    print(f"{name} Mean: {np.round(np.nanmean(values), 4)}")
    print(f"{name} Median: {np.round(np.nanmedian(values), 4)}")
    print(f"{name} Std: {np.round(np.nanstd(values), 4)}")
    print()


def split_data_name(data_name):
    parts = data_name.split("_", 1)
    dataset = parts[0]
    omics = parts[1] if len(parts) > 1 else ""
    return dataset, omics


def main(args):
    data_name = args.data_name
    miss_rate = args.miss_rate
    seeds = args.seeds
    dataset, omics = split_data_name(data_name)

    print("\n")
    print(data_name)
    print(f"Missingness: {miss_rate}")
    print(f"Seeds: {seeds}")
    if args.tuning_param:
        print(f"Tuning: {args.tuning_param} = {args.tuning_value}")
    print("\n")

    rmse_list = []
    mae_list = []
    pearson_list = []
    spearman_list = []
    runtime_list = []

    last_imputed_data_x = None

    for seed in seeds:
        print(f"Running seed: {seed}")

        # loading inside the seed loop means each seed produces a reproducible independent missingness mask.
        ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, seed=seed)

        start_time = time.time()
        imputed_data_x = train_and_impute(
            miss_data_x,
            data_m,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            noise_rate=args.noise_rate,
            dropout=args.dropout,
            seed=seed,
            device=args.device,
        )
        runtime_seconds = time.time() - start_time

        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
        mae = mae_loss(ori_data_x, imputed_data_x, data_m)
        pearson = pearson_masked(ori_data_x, imputed_data_x, data_m)
        spearman = spearman_masked(ori_data_x, imputed_data_x, data_m)

        rmse_list.append(rmse)
        mae_list.append(mae)
        pearson_list.append(pearson)
        spearman_list.append(spearman)
        runtime_list.append(runtime_seconds)
        last_imputed_data_x = imputed_data_x

        row = {
            "experiment_type": args.experiment_type,
            "model": "DAE",
            "dataset": dataset,
            "omics": omics,
            "data_name": data_name,
            "missingness": miss_rate,
            "seed": seed,
            "tuning_param": args.tuning_param,
            "tuning_value": args.tuning_value,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "noise_rate": args.noise_rate,
            "dropout": args.dropout,
            "rmse": rmse,
            "mae": mae,
            "pearson": pearson,
            "spearman": spearman,
            "runtime_seconds": runtime_seconds,
        }
        append_csv(args.output_csv, row)

        print("RMSE Performance: " + str(np.round(rmse, 4)))
        print("MAE Performance: " + str(np.round(mae, 4)))
        print("Pearson Performance: " + str(np.round(pearson, 4)))
        print("Spearman Performance: " + str(np.round(spearman, 4)))
        print("Runtime seconds: " + str(np.round(runtime_seconds, 2)))
        print(f"Loaded {args.data_name}: shape = {miss_data_x.shape}")
        print()

    print("=== Summary across seeds ===")
    summarize_metric("RMSE", rmse_list)
    summarize_metric("MAE", mae_list)
    summarize_metric("Pearson", pearson_list)
    summarize_metric("Spearman", spearman_list)
    summarize_metric("Runtime seconds", runtime_list)

    return last_imputed_data_x, {
        "rmse_mean": float(np.nanmean(rmse_list)),
        "mae_mean": float(np.nanmean(mae_list)),
        "pearson_mean": float(np.nanmean(pearson_list)),
        "spearman_mean": float(np.nanmean(spearman_list)),
        "runtime_mean": float(np.nanmean(runtime_list)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="BRCA_CNV", type=str)
    parser.add_argument("--miss_rate", default=0.3, type=float)

    # DAE params
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--noise_rate", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)

    # multiple independent runs
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])

    # recording / metadata
    parser.add_argument("--output_csv", default=None, type=str)
    parser.add_argument("--experiment_type", default="final", type=str, choices=["tuning", "final", "test"])
    parser.add_argument("--tuning_param", default="", type=str)
    parser.add_argument("--tuning_value", default="", type=str)

    # runtime/device
    parser.add_argument("--device", default=None, type=str)

    args = parser.parse_args()
    main(args)
