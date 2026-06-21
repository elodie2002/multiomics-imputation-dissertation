# metrics.py
import numpy as np
from scipy.stats import pearsonr, spearmanr

def normalization(data, parameters=None):
    """Normalize data to [0, 1] per feature, ignoring NaNs."""
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
        parameters = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

    return norm_data, parameters


def rmse_loss(ori_data, imputed_data, data_m):
    """RMSE only on originally-missing entries, excluding NaN/inf."""
    ori_data, params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, params)

    mask = (1 - data_m).astype(bool)
    valid = mask & np.isfinite(ori_data) & np.isfinite(imputed_data)

    x = ori_data[valid]
    y = imputed_data[valid]

    if len(x) == 0:
        return np.nan

    return np.sqrt(np.mean((x - y) ** 2))


def mae_loss(ori_data, imputed_data, data_m):
    """MAE only on originally-missing entries, excluding NaN/inf."""
    ori_data, params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, params)

    mask = (1 - data_m).astype(bool)
    valid = mask & np.isfinite(ori_data) & np.isfinite(imputed_data)

    x = ori_data[valid]
    y = imputed_data[valid]

    if len(x) == 0:
        return np.nan

    return np.mean(np.abs(x - y))


def binary_sampler(p, rows, cols):
    """Sample binary matrix with P(1)=p."""
    unif = np.random.uniform(0.0, 1.0, size=[rows, cols])
    return 1 * (unif < p)

def pearson_masked(ori_data, imputed_data, data_m):
    """Pearson correlation only on originally-missing entries."""
    ori_data, params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, params)

    mask = (1 - data_m).astype(bool)
    valid = mask & np.isfinite(ori_data) & np.isfinite(imputed_data)

    x = ori_data[valid]
    y = imputed_data[valid]

    # avoid failure on degenerate cases
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return pearsonr(x, y)[0]


def spearman_masked(ori_data, imputed_data, data_m):
    """Spearman correlation only on originally-missing entries."""
    ori_data, params = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, params)

    mask = (1 - data_m).astype(bool)
    valid = mask & np.isfinite(ori_data) & np.isfinite(imputed_data)

    x = ori_data[valid]
    y = imputed_data[valid]

    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return spearmanr(x, y)[0]