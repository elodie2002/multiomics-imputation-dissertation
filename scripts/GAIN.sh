#!/bin/bash
# Final GAIN benchmark script with multiple seeds.

current_dir=$(pwd)
cd ../../Baseline_and_Metric/Imputation/GAIN

output_txt="$current_dir/GAIN_final_log.txt"
output_csv="$current_dir/GAIN_final_results.csv"

seeds=(42 123 2024 3407 9999)

# Final parameters 
final_batch_size=128
final_hint_rate=0.9
final_alpha=100
final_iterations=500

# single-case mode: ./GAIN.sh BRCA mRNA 0.5
if [ $# -eq 3 ]; then
  data_name="${1}_${2}"
  miss_rate="$3"

  echo "==================================" >> "$output_txt"
  echo "GAIN final | Dataset: $data_name | Missingness: $miss_rate" >> "$output_txt"
  echo "==================================" >> "$output_txt"

  python3 main_letter_spam.py \
    --data_name "$data_name" \
    --miss_rate "$miss_rate" \
    --seeds "${seeds[@]}" \
    --experiment_type final \
    --tuning_param final \
    --tuning_value final \
    --output_csv "$output_csv" \
    --batch_size "$final_batch_size" \
    --hint_rate "$final_hint_rate" \
    --alpha "$final_alpha" \
    --iterations "$final_iterations" \
    >> "$output_txt" 2>&1

  cd "$current_dir"
  exit 0
fi

> "$output_txt"
> "$output_csv"

datasets=("BRCA" "COAD" "GBM" "LGG" "OV")
omics=("CNV" "Methy" "miRNA" "mRNA")
miss_rates=("0.3" "0.5" "0.7")

for dataset in "${datasets[@]}"
do
  for omic in "${omics[@]}"
  do
    data_name="${dataset}_${omic}"

    for miss_rate in "${miss_rates[@]}"
    do
      echo "==================================" >> "$output_txt"
      echo "GAIN final | Dataset: $data_name | Missingness: $miss_rate" >> "$output_txt"
      echo "==================================" >> "$output_txt"

      python3 main_letter_spam.py \
        --data_name "$data_name" \
        --miss_rate "$miss_rate" \
        --seeds "${seeds[@]}" \
        --experiment_type final \
        --tuning_param final \
        --tuning_value final \
        --output_csv "$output_csv" \
        --batch_size "$final_batch_size" \
        --hint_rate "$final_hint_rate" \
        --alpha "$final_alpha" \
        --iterations "$final_iterations" \
        >> "$output_txt" 2>&1

      echo "" >> "$output_txt"
    done
  done
done

cd "$current_dir"
