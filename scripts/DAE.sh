#!/bin/bash
# Final DAE full benchmark script after hyperparameters are frozen.

current_dir=$(pwd)
cd ../../Baseline_and_Metric/Imputation/DAE

output_txt="$current_dir/DAE_final_log.txt"
output_csv="$current_dir/DAE_final_results.csv"

> "$output_txt"
> "$output_csv"

datasets=("BRCA" "COAD" "GBM" "LGG" "OV")
omics=("CNV" "Methy" "miRNA" "mRNA")
miss_rates=("0.3" "0.5" "0.7")
seeds=(42 123 2024 3407 9999)

for dataset in "${datasets[@]}"
do
  for omic in "${omics[@]}"
  do
    data_name="${dataset}_${omic}"

    for miss_rate in "${miss_rates[@]}"
    do
      echo "==================================" >> "$output_txt"
      echo "DAE final | Dataset: $data_name | Missingness: $miss_rate" >> "$output_txt"
      echo "==================================" >> "$output_txt"

      python3 main_dae.py \
        --data_name "$data_name" \
        --miss_rate "$miss_rate" \
        --seeds "${seeds[@]}" \
        --experiment_type final \
        --output_csv "$output_csv" \
        --hidden_dim 256 \
        --epochs 100 \
        --noise_rate 0.1 \
        --dropout 0.2 \
        --tuning_param "final" \
        --tuning_value "final"
        >> "$output_txt"

      echo "" >> "$output_txt"
    done
  done
done

cd "$current_dir"
