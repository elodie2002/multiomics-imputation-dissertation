#!/bin/bash
# GAIN fair sensitivity analysis.
# Representative cases + multiple missingness levels + multiple seeds.

current_dir=$(pwd)
cd ../../Baseline_and_Metric/Imputation/GAIN || exit 1

output_txt="$current_dir/GAIN_tuning_log.txt"
output_csv="$current_dir/GAIN_tuning_results.csv"

> "$output_txt"
> "$output_csv"

cases=("BRCA_mRNA" "LGG_CNV" "OV_miRNA")
miss_rates=("0.3" "0.5" "0.7")
seeds=(42 123 2024)

total_groups=$(( ${#cases[@]} * ${#miss_rates[@]} * 7 ))
current_group=0

run_case () {
  local data_name=$1
  local miss_rate=$2
  local tuning_param=$3
  local tuning_value=$4
  shift 4

  current_group=$((current_group + 1))

  echo "[Group $current_group / $total_groups] GAIN | $data_name | missingness=$miss_rate | $tuning_param=$tuning_value"

  echo "==================================" >> "$output_txt"
  echo "GAIN tuning | Dataset: $data_name | Missingness: $miss_rate | $tuning_param=$tuning_value" >> "$output_txt"
  echo "==================================" >> "$output_txt"

  python3 main_letter_spam.py \
    --data_name "$data_name" \
    --miss_rate "$miss_rate" \
    --seeds "${seeds[@]}" \
    --experiment_type tuning \
    --tuning_param "$tuning_param" \
    --tuning_value "$tuning_value" \
    --output_csv "$output_csv" \
    "$@" \
    >> "$output_txt" 2>&1

  echo "" >> "$output_txt"
}

for data_name in "${cases[@]}"
do
  for miss_rate in "${miss_rates[@]}"
  do
    # Baseline: iterations=500, alpha=100, hint_rate=0.9, batch_size=128

    for iterations in 100 500 1000
    do
      run_case "$data_name" "$miss_rate" "iterations" "$iterations" \
        --iterations "$iterations" --alpha 100 --hint_rate 0.9 --batch_size 128
    done

    for alpha in 10 100
    do
      run_case "$data_name" "$miss_rate" "alpha" "$alpha" \
        --iterations 500 --alpha "$alpha" --hint_rate 0.9 --batch_size 128
    done

    for hint_rate in 0.7 0.9
    do
      run_case "$data_name" "$miss_rate" "hint_rate" "$hint_rate" \
        --iterations 500 --alpha 100 --hint_rate "$hint_rate" --batch_size 128
    done
  done
done

cd "$current_dir" || exit 1