#!/bin/bash

current_dir=$(pwd) 
cd ../../Baseline_and_Metric/Imputation/VAE || exit 1

output_txt="$current_dir/VAE_tuning_log.txt"
output_csv="$current_dir/VAE_tuning_results.csv"

> "$output_txt"
> "$output_csv"

cases=("BRCA_mRNA" "LGG_CNV" "OV_miRNA")
miss_rates=("0.3" "0.5" "0.7")
seeds=(42 123 2024)

total_groups=$(( ${#cases[@]} * ${#miss_rates[@]} * 9 ))
current_group=0

run_case () {
  local data_name=$1
  local miss_rate=$2
  local tuning_param=$3
  local tuning_value=$4
  shift 4

  current_group=$((current_group + 1))

  echo "[Group $current_group / $total_groups] VAE | $data_name | missingness=$miss_rate | $tuning_param=$tuning_value"

  echo "==================================" >> "$output_txt"
  echo "VAE tuning | Dataset: $data_name | Missingness: $miss_rate | $tuning_param=$tuning_value" >> "$output_txt"
  echo "==================================" >> "$output_txt"

  python3 main_vae.py \
    --data_name "$data_name" \
    --miss_rate "$miss_rate" \
    --seeds "${seeds[@]}" \
    --experiment_type tuning \
    --tuning_param "$tuning_param" \
    --tuning_value "$tuning_value" \
    --output_csv "$output_csv" \
    "$@" \
    >> "$output_txt"

  echo "" >> "$output_txt"
}

for data_name in "${cases[@]}"
do
  for miss_rate in "${miss_rates[@]}"
  do
    # Baseline VAE configuration:
    # hidden_dim=256, latent_dim=32, epochs=100, noise_rate=0.1, beta=1e-6, dropout=0.1

    # 1) latent_dim sensitivity
    for latent_dim in 16 32 64
    do
      run_case "$data_name" "$miss_rate" "latent_dim" "$latent_dim" \
        --hidden_dim 256 --latent_dim "$latent_dim" --epochs 100 --noise_rate 0.1 --beta 1e-6 --dropout 0.1
    done

    # 2) beta sensitivity
    for beta in 1e-6 1e-5
    do
      run_case "$data_name" "$miss_rate" "beta" "$beta" \
        --hidden_dim 256 --latent_dim 32 --epochs 100 --noise_rate 0.1 --beta "$beta" --dropout 0.1
    done

    # 3) epochs sensitivity
    for epochs in 100 200
    do
      run_case "$data_name" "$miss_rate" "epochs" "$epochs" \
        --hidden_dim 256 --latent_dim 32 --epochs "$epochs" --noise_rate 0.1 --beta 1e-6 --dropout 0.1
    done

    # 4) dropout sensitivity
    for dropout in 0.1 0.2
    do
      run_case "$data_name" "$miss_rate" "dropout" "$dropout" \
        --hidden_dim 256 --latent_dim 32 --epochs 100 --noise_rate 0.1 --beta 1e-6 --dropout "$dropout"
    done
  done
done

cd "$current_dir" || exit 1