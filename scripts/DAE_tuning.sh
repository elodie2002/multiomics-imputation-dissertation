#!/bin/bash
# DAE sensitivity analysis.

current_dir=$(pwd)
cd ../../Baseline_and_Metric/Imputation/DAE

output_txt="$current_dir/DAE_tuning_log.txt"
output_csv="$current_dir/DAE_tuning_results.csv"

> "$output_txt"
> "$output_csv"

# Representative tuning subset: medium, easy, hard cases
cases=("BRCA_mRNA" "LGG_CNV" "OV_miRNA")
miss_rates=("0.3" "0.5" "0.7")
seeds=(42 123 2024)

run_case () {
  local data_name=$1
  local miss_rate=$2
  local tuning_param=$3
  local tuning_value=$4
  shift 4

  echo "==================================" >> "$output_txt"
  echo "DAE tuning | Dataset: $data_name | Missingness: $miss_rate | $tuning_param=$tuning_value" >> "$output_txt"
  echo "==================================" >> "$output_txt"

  python3 main_dae.py \
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
    # 1) hidden_dim sensitivity; keep other parameters fixed
    for hidden_dim in 128 256 512
    do
      run_case "$data_name" "$miss_rate" "hidden_dim" "$hidden_dim" \
        --hidden_dim "$hidden_dim" --epochs 100 --noise_rate 0.1 --dropout 0.2
    done

    # 2) epochs sensitivity; use hidden_dim=256 as baseline
    for epochs in 100 200
    do
      run_case "$data_name" "$miss_rate" "epochs" "$epochs" \
        --hidden_dim 256 --epochs "$epochs" --noise_rate 0.1 --dropout 0.2
    done

    # 3) noise_rate sensitivity; use hidden_dim=256, epochs=100 as baseline
    for noise_rate in 0.0 0.1 0.2
    do
      run_case "$data_name" "$miss_rate" "noise_rate" "$noise_rate" \
        --hidden_dim 256 --epochs 100 --noise_rate "$noise_rate" --dropout 0.2
    done

    # 4) dropout sensitivity; for DAE stability
    for dropout in 0.1 0.2 0.3
    do
      run_case "$data_name" "$miss_rate" "dropout" "$dropout" \
        --hidden_dim 256 --epochs 100 --noise_rate 0.1 --dropout "$dropout"
    done
  done
done

cd "$current_dir"