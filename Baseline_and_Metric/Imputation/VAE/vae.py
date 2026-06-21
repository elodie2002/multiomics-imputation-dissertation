from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        enc_mid = max(hidden_dim // 2, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_layer = nn.Linear(enc_mid, latent_dim)
        self.logvar_layer = nn.Linear(enc_mid, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, enc_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_mid, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def train_and_impute(
    miss_data_x: np.ndarray,
    data_m: np.ndarray,
    *,
    hidden_dim: int = 256,
    latent_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    epochs: int = 200,
    noise_rate: float = 0.1,
    beta: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Train a VAE on miss_data_x and return the imputed matrix.

    The reconstruction loss is calculated only on observed entries.
    Extra denoising corruption is applied only to observed entries.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = miss_data_x.astype(np.float32)
    M = data_m.astype(np.float32)

    X_filled = np.nan_to_num(X, nan=0.0)

    n, dim = X_filled.shape
    model = VAE(
        dim=dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    X_t = torch.from_numpy(X_filled).to(device)
    M_t = torch.from_numpy(M).to(device)

    model.train()
    for _ in range(epochs):
        idx = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X_t[batch_idx]
            mb = M_t[batch_idx]

            if noise_rate > 0:
                noise_mask = (torch.rand_like(mb) > noise_rate).float()
                effective_mask = mb * noise_mask
            else:
                effective_mask = mb

            x_corrupt = xb * effective_mask

            recon, mu, logvar = model(x_corrupt)

            recon_loss = (((recon - xb) ** 2) * mb).sum() / (mb.sum() + 1e-8)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t)
        recon_all = model.decode(mu).cpu().numpy()

    imputed = X_filled.copy()
    imputed[M == 0] = recon_all[M == 0]

    return imputed
