# data_loader.py
from typing import Optional
import numpy as np
from metrics import binary_sampler


def data_loader(data_name: str, miss_rate: float, seed: Optional[int] = None):
    """
    Load complete CSV data and introduce artificial missingness.

    Args:
        data_name: dataset name, e.g. BRCA_mRNA
        miss_rate: probability of missing entries
        seed: optional random seed used for reproducible mask generation

    Returns:
        data_x: original complete data
        miss_data_x: data with artificial missing values
        data_m: mask matrix, 1 = observed, 0 = missing
    """
    if seed is not None:
        np.random.seed(seed)

    file_name = (
        "../../../Main_Dataset/Imputation_datasets/Imp-"
        + data_name.split("_")[0]
        + "/Top/"
        + data_name
        + ".csv"
    )

    raw = np.genfromtxt(file_name, delimiter=",", dtype=str)
    data_x = raw[1:, 1:]
    data_x[data_x == ""] = np.nan
    data_x = data_x.astype(float)

    no, dim = data_x.shape

    # Binary mask: 1 = observed, 0 = artificially missing
    data_m = binary_sampler(1 - miss_rate, no, dim)

    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m