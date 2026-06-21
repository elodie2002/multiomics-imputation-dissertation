#!/bin/bash
# VAE final benchmark script with multiple seeds and CSV output.

current_dir=$(pwd)
cd ../../Baseline_and_Metric/Imputation/VAE || exit 1

output_txt="$current_dir/VAE_final_log.txt"
output_csv="$current_dir/VAE_final_results.csv"

seeds=(42 123 2024 3407 9999)

# Final hyperparameters 
hidden_dim=256
latent_dim=32
epochs=100
lr=1e-3
batch_size=128
noise_rate=0.1
beta=1e-6
dropout=0.1

# single-case mode: ./VAE.sh BRCA mRNA 0.5
if [ $# -eq 3 ]; then
  dataset=$1
  omic=$2
  miss_rate=$3
  data_name="${dataset}_${omic}"

  echo "==================================" >> "$output_txt"
  echo "VAE final | Dataset: $data_name | Missingness: $miss_rate" >> "$output_txt"
  echo "==================================" >> "$output_txt"

  python3 main_vae.py \
    --data_name "$data_name" \
    --miss_rate "$miss_rate" \
    --seeds "${seeds[@]}" \
    --experiment_type final \
    --tuning_param final \
    --tuning_value final \
    --output_csv "$output_csv" \
    --epochs "$epochs" \
    --hidden_dim "$hidden_dim" \
    --latent_dim "$latent_dim" \
    --lr "$lr" \
    --batch_size "$batch_size" \
    --noise_rate "$noise_rate" \
    --beta "$beta" \
    --dropout "$dropout" \
    >> "$output_txt" 2>&1

  cd "$current_dir" || exit 1
  exit 0
fi

> "$output_txt"
> "$output_csv"

datasets=("BRCA" "COAD" "GBM" "LGG" "OV")
omics=("CNV" "Methy" "miRNA" "mRNA")
miss_rates=("0.3" "0.5" "0.7")

total_groups=$(( ${#datasets[@]} * ${#omics[@]} * ${#miss_rates[@]} ))
current_group=0

for dataset in "${datasets[@]}"
do
  for omic in "${omics[@]}"
  do
    data_name="${dataset}_${omic}"

    for miss_rate in "${miss_rates[@]}"
    do
      current_group=$((current_group + 1))
      echo "[Group $current_group / $total_groups] VAE final | $data_name | missingness=$miss_rate"

      echo "==================================" >> "$output_txt"
      echo "VAE final | Dataset: $data_name | Missingness: $miss_rate" >> "$output_txt"
      echo "==================================" >> "$output_txt"

      python3 main_vae.py \
        --data_name "$data_name" \
        --miss_rate "$miss_rate" \
        --seeds "${seeds[@]}" \
        --experiment_type final \
        --tuning_param final \
        --tuning_value final \
        --output_csv "$output_csv" \
        --epochs "$epochs" \
        --hidden_dim "$hidden_dim" \
        --latent_dim "$latent_dim" \
        --lr "$lr" \
        --batch_size "$batch_size" \
        --noise_rate "$noise_rate" \
        --beta "$beta" \
        --dropout "$dropout" \
        >> "$output_txt" 2>&1

      echo "" >> "$output_txt"
    done
  done
done

cd "$current_dir" || exit 1