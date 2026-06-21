# dae.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

# Inherits from PyTorch neural network module
class DAE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

#main function 
def train_and_impute(
    miss_data_x: np.ndarray,
    data_m: np.ndarray,
    *,
    # some Default hyperparameters
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    noise_rate: float = 0.1,
    dropout: float = 0.2,
    seed: int = 42,
    device: str | None = None,
) -> np.ndarray:
    """
    Trains a denoising autoencoder on miss_data_x.
    - Uses ONLY observed entries (mask==1) in loss.
    - Adds extra random masking noise to observed entries during training (noise_rate).
    - After training: imputes missing entries and returns full imputed matrix.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = miss_data_x.astype(np.float32)
    M = data_m.astype(np.float32)  # 1 observed, 0 missing

    # Fill NaNs with 0 for network input, but keep mask to ignore in loss
    X_filled = np.nan_to_num(X, nan=0.0)

    #n = number of samples (rows), dim = number of features (columns)
    n, dim = X_filled.shape
    model = DAE(dim=dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X_filled).to(device)
    M_t = torch.from_numpy(M).to(device)

    model.train()
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = X_t[batch_idx] 
            mb = M_t[batch_idx]

            # Denoising: additionally drop some observed entries
            if noise_rate > 0:
                noise_mask = (torch.rand_like(mb) > noise_rate).float()
                # keep original missing as missing, only further corrupt observed
                effective_mask = mb * noise_mask
            else:
                effective_mask = mb

            # input corruption: zero out dropped entries
            x_corrupt = xb * effective_mask

            recon = model(x_corrupt)

            # loss only on originally observed entries (mb==1)
            loss = (((recon - xb) ** 2) * mb).sum() / (mb.sum() + 1e-8)

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Inference: reconstruct from observed + zeros for missing
    model.eval()
    with torch.no_grad():
        recon_all = model(X_t).cpu().numpy()

    # Impute missing entries with reconstruction, keep observed as original
    imputed = X_filled.copy()
    imputed[M == 0] = recon_all[M == 0]
    return imputed
