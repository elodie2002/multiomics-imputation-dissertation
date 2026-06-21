# coding=utf-8
"""Main function for GAIN imputation experiments."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from data_loader import data_loader
from gain import gain
from utils import rmse_loss, mae_loss, pearson_masked, spearman_masked


CSV_COLUMNS = [
    'experiment_type', 'model', 'dataset', 'omics', 'data_name', 'miss_rate', 'seed',
    'tuning_param', 'tuning_value', 'batch_size', 'hint_rate', 'alpha', 'iterations',
    'rmse', 'mae', 'pearson', 'spearman', 'runtime_seconds'
]


def parse_seeds(seeds):
    parsed = []
    for item in seeds:
        for part in str(item).split(','):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def append_csv(path, row):
    if not path:
        return
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return np.nanmean(values), np.nanmedian(values), np.nanstd(values)


def main(args):
    data_name = args.data_name
    miss_rate = args.miss_rate
    seeds = parse_seeds(args.seeds)

    dataset = data_name.split('_')[0]
    omics = data_name.split('_', 1)[1] if '_' in data_name else ''

    print('\n')
    print(data_name)
    print('Missingness:', miss_rate)
    print('Seeds:', seeds)
    if args.experiment_type == 'tuning':
        print('Tuning:', args.tuning_param, '=', args.tuning_value)
    print('\n')

    all_metrics = {
        'rmse': [],
        'mae': [],
        'pearson': [],
        'spearman': [],
        'runtime_seconds': [],
    }

    for seed in seeds:
        print('Running seed:', seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, seed=seed)

        gain_parameters = {
            'batch_size': args.batch_size,
            'hint_rate': args.hint_rate,
            'alpha': args.alpha,
            'iterations': args.iterations,
            'seed': seed,
        }

        start_time = time.time()
        imputed_data_x = gain(miss_data_x, gain_parameters)
        runtime_seconds = time.time() - start_time

        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
        mae = mae_loss(ori_data_x, imputed_data_x, data_m)
        pearson = pearson_masked(ori_data_x, imputed_data_x, data_m)
        spearman = spearman_masked(ori_data_x, imputed_data_x, data_m)

        all_metrics['rmse'].append(rmse)
        all_metrics['mae'].append(mae)
        all_metrics['pearson'].append(pearson)
        all_metrics['spearman'].append(spearman)
        all_metrics['runtime_seconds'].append(runtime_seconds)

        print('RMSE Performance:', np.round(rmse, 4))
        print('MAE Performance:', np.round(mae, 4))
        print('Pearson Performance:', np.round(pearson, 4))
        print('Spearman Performance:', np.round(spearman, 4))
        print('Runtime seconds:', np.round(runtime_seconds, 2))

        append_csv(args.output_csv, {
            'experiment_type': args.experiment_type,
            'model': 'GAIN',
            'dataset': dataset,
            'omics': omics,
            'data_name': data_name,
            'miss_rate': miss_rate,
            'seed': seed,
            'tuning_param': args.tuning_param,
            'tuning_value': args.tuning_value,
            'batch_size': args.batch_size,
            'hint_rate': args.hint_rate,
            'alpha': args.alpha,
            'iterations': args.iterations,
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman,
            'runtime_seconds': runtime_seconds,
        })

    print('\n=== Summary across seeds ===')
    for metric, values in all_metrics.items():
        mean, median, std = summarize(values)
        print(metric.upper(), 'Mean:', np.round(mean, 4))
        print(metric.upper(), 'Median:', np.round(median, 4))
        print(metric.upper(), 'Std:', np.round(std, 4))
        print()

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='BRCA_mRNA', type=str)
    parser.add_argument('--miss_rate', default=0.5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hint_rate', default=0.9, type=float)
    parser.add_argument('--alpha', default=100, type=float)
    parser.add_argument('--iterations', default=500, type=int)
    parser.add_argument('--seeds', nargs='+', default=['42'], help='Seeds, e.g. --seeds 42 123 2024')
    parser.add_argument('--experiment_type', default='final', choices=['tuning', 'final'])
    parser.add_argument('--tuning_param', default='final')
    parser.add_argument('--tuning_value', default='final')
    parser.add_argument('--output_csv', default=None)

    args = parser.parse_args()
    main(args)