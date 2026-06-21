# Cancer Multi-Omics Imputation Benchmark

This repository contains the code, model implementations, result files, and analysis scripts used for the dissertation project:

**Benchmarking Deep-Learning Methods for Cancer Multi-Omics Imputation**

## Project Overview

This project builds on the publicly available **MLOmics: Cancer Multi-Omics Database for Machine Learning** benchmark framework and focuses specifically on the **multi-omics imputation task**.

The aim of this project is to compare three deep-learning imputation models under a shared benchmark setting:

* Denoising Autoencoder (DAE)
* Variational Autoencoder (VAE)
* Generative Adversarial Imputation Nets (GAIN)

The models are evaluated across shared datasets, omics modalities, missingness levels, random seeds, and evaluation metrics. The analysis includes:

* reconstruction-error metrics;
* correlation-based metrics;
* runtime comparison;
* hyperparameter tuning;
* robustness analysis;
* dataset–omics interaction analysis;
* failure-localisation analysis.

## Relationship to MLOmics

This repository builds on the public MLOmics benchmark repository by Yang et al.:

https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

The original MLOmics repository provides the benchmark framework, processed TCGA multi-omics datasets, imputation task structure, and the original GAIN implementation used as the starting point for this project.

This repository does **not** introduce a new cancer multi-omics dataset. Instead, it extends the MLOmics imputation setting into a focused benchmark comparing deep-learning imputation methods.

## Components Based on the Original MLOmics Framework

The following components are based on, or follow, the original MLOmics repository:

* processed TCGA multi-omics imputation dataset structure;
* benchmark folder organisation, including `Main_Dataset/` and parts of `Baseline_and_Metric/`;
* original GAIN baseline code and related utility functions;
* original benchmark conventions for dataset names, omics modalities, and missingness settings.

Original licence headers have been preserved where applicable.

## Contributions of This Repository

The main additions and modifications in this repository are:

* implementation of the Denoising Autoencoder (DAE) imputation model;
* implementation of the Variational Autoencoder (VAE) imputation model;
* adaptation and tuning of the GAIN baseline under the same benchmark conditions;
* automation scripts for DAE, VAE, and GAIN tuning and final benchmark execution;
* extension of the evaluation pipeline to include Pearson correlation, Spearman correlation, and runtime;
* refinement of RMSE and MAE evaluation so that metrics are calculated on valid artificially masked entries;
* result aggregation scripts combining outputs across datasets, omics modalities, missingness levels, and random seeds;
* figure-generation scripts for tuning summaries, model comparison, robustness analysis, dataset–omics interaction analysis, and failure-localisation analysis;
* final CSV result files used to generate the dissertation findings.

In summary, MLOmics provides the benchmark foundation and original GAIN baseline, while this repository contributes additional neural model implementations, controlled multi-run execution, extended evaluation metrics, result aggregation, and analysis scripts.

## Repository Structure

```text
cancer-multi-omics-imputation-benchmark/
│
├── README.md
├── environment.yml
│
├── Baseline_and_Metric/
│   └── Imputation/
│       ├── DAE/
│       ├── VAE/
│       └── GAIN/
│
├── Main_Dataset/
│   └── Imputation_datasets/   # Download from original MLOmics repository if not included
│
└── scripts/
    ├── DAE.sh
    ├── VAE.sh
    ├── GAIN.sh
    ├── DAE_tuning.sh
    ├── VAE_tuning.sh
    ├── GAIN_tuning.sh
    │
    ├── plot_final_results.py
    ├── plot_further_analysis.py
    ├── plot_remove_severe_cases.py
    └── plot_tuning_summary.py
```

## Data Availability and Dataset Setup

The processed TCGA multi-omics imputation datasets used in this project are based on the publicly available MLOmics benchmark repository:

https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

Due to the size and original ownership of the benchmark data, the dataset files may not be fully included in this repository. To reproduce the benchmark experiments, download the required imputation datasets from the original MLOmics repository and place them in the following directory:

```text
Main_Dataset/Imputation_datasets/
```

The expected structure is:

```text
cancer-multi-omics-imputation-benchmark/
└── Main_Dataset/
    └── Imputation_datasets/
```

The scripts in `scripts/` assume that the dataset is available at this relative path. If the dataset is stored elsewhere, the dataset path in the corresponding model scripts may need to be updated.

This repository does not claim ownership of the original MLOmics datasets. The datasets remain part of the original MLOmics benchmark framework by Yang et al.

## Environment

The experiments were run in a Conda environment using Python 3.9.

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate mlomics
```

## Running the Benchmark

The main benchmark scripts are located in the `scripts/` folder.

Final benchmark commands:

```bash
cd scripts

bash DAE.sh
bash VAE.sh
bash GAIN.sh
```

Hyperparameter tuning scripts:

```bash
bash DAE_tuning.sh
bash VAE_tuning.sh
bash GAIN_tuning.sh
```

These scripts run the corresponding model experiments and produce CSV result files.

## Result and Figure Generation

The analysis and plotting scripts are also located in the `scripts/` folder.

Main result plots:

```bash
python plot_final_results.py
```

Additional setting-level and interaction analysis plots:

```bash
python plot_further_analysis.py
```

Sensitivity analysis after removing severe cases:

```bash
python plot_remove_severe_cases.py
```

Tuning summary plots:

```bash
python plot_tuning_summary.py
```

## Result Files

Result CSV files are not included in this public repository. They can be regenerated by running the benchmark scripts and analysis scripts described above.

The generated outputs include final benchmark results, hyperparameter tuning results, aggregated analysis files, and figure-generation inputs.

## Notes on Reproducibility

The benchmark results may vary slightly depending on hardware, package versions, and random seed behaviour. The experiments were designed to reduce this variation by using shared benchmark settings, fixed random seeds, and consistent evaluation metrics across models.

## Attribution

This project builds on the MLOmics benchmark framework:

**MLOmics: Cancer Multi-Omics Database for Machine Learning**
Original repository: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

The original benchmark framework, dataset structure, and GAIN baseline are attributed to the original MLOmics authors. This repository provides dissertation-specific extensions, additional model implementations, evaluation refinements, result aggregation, and analysis scripts.

## Licence

Please refer to the original MLOmics repository for the licence terms covering the original benchmark framework and dataset-related components.

Any additional code written specifically for this repository is provided for research and educational purposes. Before reusing or redistributing this repository, please check the licence conditions of the original MLOmics project.

