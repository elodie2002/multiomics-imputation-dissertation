# data_loader.py
import numpy as np
from metrics import binary_sampler
from typing import Optional

def data_loader(data_name: str, miss_rate: float, seed: Optional[int] = None):
    """
    Load complete CSV data and introduce synthetic missingness.

    Parameters
    ----------
    data_name : str
        Dataset name, e.g. BRCA_mRNA.
    miss_rate : float
        Probability of masking an entry.
    seed : int | None
        Random seed used for mask generation. If provided, each seed creates
        a reproducible missingness mask.

    Returns
    -------
    data_x : np.ndarray
        Complete original data matrix, used only as evaluation reference.
    miss_data_x : np.ndarray
        Input matrix with masked entries replaced by NaN.
    data_m : np.ndarray
        Binary mask, where 1 = observed and 0 = masked/missing.
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

    # 1 = observed, 0 = artificially masked
    data_m = binary_sampler(1 - miss_rate, no, dim)

    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m
