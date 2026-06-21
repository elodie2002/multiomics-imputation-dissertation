# Dissertation Code Submission

This code submission contains the scripts, model implementations, result files, and analysis code used for the dissertation:

**Benchmarking Deep-Learning Methods for Cancer Multi-Omics Imputation**

## Project Overview

This project builds on the publicly available MLOmics benchmark framework for cancer multi-omics analysis. The dissertation focuses on the imputation task and compares three deep-learning imputation models under a shared benchmark setting:

- Denoising Autoencoder (DAE)
- Variational Autoencoder (VAE)
- Generative Adversarial Imputation Nets (GAIN)

The models were evaluated across shared datasets, omics modalities, missingness levels, random seeds, and evaluation metrics. The analysis includes reconstruction-error metrics, correlation-based metrics, runtime comparison, hyperparameter tuning, and failure-localisation analysis.

## Attribution and Scope of Modifications

This code submission builds on the public **MLOmics: Cancer Multi-Omics Database for Machine Learning** repository by Yang et al. The original MLOmics repository provides the benchmark framework, processed TCGA multi-omics datasets, imputation task structure, and the original GAIN implementation used as the starting point for this dissertation. The original repository is available at: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark

The work submitted here does not introduce a new cancer multi-omics dataset. Instead, it extends the MLOmics imputation setting into a focused dissertation benchmark of deep-learning imputation models.

### Components from the original MLOmics framework

The following components are based on, or follow, the original MLOmics repository:

- processed TCGA multi-omics imputation dataset structure;
- benchmark folder organisation, including `Main_Dataset/` and parts of `Baseline_and_Metric/`;
- original GAIN baseline code and related utility functions;
- original benchmark conventions for dataset names, omics modalities, and missingness settings.

Original licence headers have been preserved where applicable.

### Contributions added or modified for this dissertation

The main dissertation-specific additions and modifications are:

- implementation of the Denoising Autoencoder (DAE) imputation model;
- implementation of the Variational Autoencoder (VAE) imputation model;
- adaptation and tuning of the GAIN baseline under the same benchmark conditions;
- automation scripts for DAE, VAE, and GAIN tuning and final benchmark execution;
- extension of the evaluation pipeline to include Pearson correlation, Spearman correlation, and runtime;
- refinement of RMSE and MAE evaluation so that metrics are calculated on valid artificially masked entries;
- result aggregation scripts used to combine outputs across datasets, omics modalities, missingness levels, and random seeds;
- figure-generation scripts for tuning summaries, model comparison, robustness analysis, dataset--omics interaction analysis, and failure-localisation analysis;
- final CSV result files used to generate the dissertation findings.

In summary, MLOmics provides the benchmark foundation and original GAIN baseline, while this dissertation contributes the additional neural model implementations, controlled multi-run execution, extended evaluation metrics, result aggregation, and analysis scripts used for the reported comparison of DAE, VAE, and GAIN.

## Folder Structure

```text
dissertation_code_submission/
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
│   └── Imputation_datasets/
│
├── final_analysis_csv/
│
├── final_deeper_insight_csv/
│
└── scripts/
    ├── DAE.sh
    ├── VAE.sh
    ├── GAIN.sh
    ├── DAE_tuning.sh
    ├── VAE_tuning.sh
    ├── GAIN_tuning.sh
    │
    ├── DAE_final_results.csv
    ├── VAE_final_results.csv
    ├── GAIN_final_results.csv
    │
    ├── DAE_tuning_results.csv
    ├── VAE_tuning_results.csv
    ├── GAIN_tuning_results.csv
    │
    ├── plot_final_results.py
    ├── plot_further_analysis.py
    ├── plot_remove_severe_cases.py
    ├── plot_tuning_summary.py
    │
    └── rmse_summary_excluding_brca_coad_mirna_high_missingness.csv
```

## Environment

The experiments were run in a Conda environment using Python 3.9.
To recreate the environment:

```bash
conda env create -f environment.yml
conda activate mlomics
```

## Running the Benchmark
The main benchmark scripts are located in the scripts/ folder

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

The analysis and plotting scripts are also located in the scripts/ folder.

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

