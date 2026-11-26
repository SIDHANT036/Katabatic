"""Utility classes for the Tabddpm model (local fallback).

This module provides compact stand-ins for the external `tabddpm` package so
that the Tabddpm model can run without that dependency (the package is not
installable from PyPI). It is a real, if small, denoising diffusion model:

- MLPDiffusion: an MLP denoiser conditioned on the diffusion step and class label
- GaussianMultinomialDiffusion: DDPM over [numerical | one-hot categorical]
  features. Numerical columns use noise prediction (MSE); each categorical
  column is a +/-1 one-hot block whose clean value is predicted with a
  cross-entropy loss, and categorical values are decoded with argmax.

If the real `tabddpm` package is installed, `models.py` will prefer that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPDiffusion(nn.Module):
    """MLP denoiser conditioned on the diffusion step and, optionally, the class.

    Parameters
    ----------
    d_in : int
        Input dimensionality (numerical features plus one-hot categorical blocks).
    num_classes : int
        Number of classes for optional y-conditioning (0 disables conditioning).
    is_y_cond : bool
        Whether to condition the denoiser on y labels.
    rtdl_params : dict
        Contains 'd_layers' (list of hidden widths) and 'dropout' (float).
    dim_t : int
        Width of the sinusoidal step embedding.
    """

    def __init__(
        self,
        d_in: int,
        num_classes: int,
        is_y_cond: bool,
        rtdl_params: dict,
        dim_t: int = 128,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.is_y_cond = bool(is_y_cond)
        self.num_classes = int(num_classes) if num_classes else 0
        self.dim_t = int(dim_t)

        hidden = list(rtdl_params.get("d_layers", [256, 256]))
        dropout = float(rtdl_params.get("dropout", 0.0))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t),
        )
        cond_dim = self.num_classes if self.is_y_cond and self.num_classes > 0 else 0
        layers: list[nn.Module] = []
        in_dim = self.d_in + self.dim_t + cond_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.d_in))
        self.net = nn.Sequential(*layers)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim_t // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim_t:
            emb = F.pad(emb, (0, self.dim_t - emb.shape[1]))
        return self.time_mlp(emb)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        parts = [x, self._time_embedding(t)]
        if self.is_y_cond and self.num_classes > 0 and y is not None:
            parts.append(
                F.one_hot(y.long().view(-1), num_classes=self.num_classes).float()
            )
        elif self.is_y_cond and self.num_classes > 0:
            parts.append(torch.zeros(x.shape[0], self.num_classes, device=x.device))
        return self.net(torch.cat(parts, dim=1))


class GaussianMultinomialDiffusion(nn.Module):
    """Denoising diffusion over [numerical | one-hot categorical] features.

    Inputs and outputs of ``mixed_loss`` and ``sample_all`` keep the layout used
    by ``models.py``: numerical columns first, then one integer index per
    categorical column. Internally each categorical column becomes a +/-1
    one-hot block, so the denoiser sees ``n_num + sum(K)`` dimensions.

    - mixed_loss(xb, out) -> (loss_multi, loss_gauss)
    - sample_all(num_samples, batch_size, y_dist, ddim=False) -> (x, y)
    """

    SAMPLE_BATCH = 4096

    def __init__(
        self,
        num_classes: Sequence[int] | np.ndarray,
        num_numerical_features: int,
        denoise_fn: nn.Module,
        gaussian_loss_type: str = "mse",
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self._denoise_fn = denoise_fn
        self.register_buffer("_dummy", torch.zeros(1))  # for .to(device)

        self._K = (
            np.array(num_classes, dtype=int)
            if len(num_classes) > 0
            else np.array([0], dtype=int)
        )
        self._cat_sizes = [int(k) for k in self._K if k > 0]
        self._n_num = int(num_numerical_features)
        self._n_cat = len(self._cat_sizes)
        self._gauss_loss = gaussian_loss_type
        self._steps = int(num_timesteps)
        self._scheduler = scheduler

        betas = self._make_betas(self._steps, scheduler)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", alphas)
        self.register_buffer("_alpha_bar", alpha_bar)

    @staticmethod
    def _make_betas(steps: int, scheduler: str) -> torch.Tensor:
        if scheduler == "linear":
            return torch.linspace(1e-4, 0.02, steps, dtype=torch.float32)
        s = 0.008
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f = torch.cos(((grid / steps) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999).float()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._denoise_fn(x, t, y)

    def _cat_slices(self):
        start = self._n_num
        for k in self._cat_sizes:
            yield slice(start, start + k)
            start += k

    def _to_internal(self, xb: torch.Tensor) -> torch.Tensor:
        """[num | cat indices] -> [num | +/-1 one-hot blocks]."""
        parts = [xb[:, : self._n_num]]
        for j, k in enumerate(self._cat_sizes):
            idx = xb[:, self._n_num + j].long().clamp(0, k - 1)
            parts.append(F.one_hot(idx, num_classes=k).float() * 2.0 - 1.0)
        return torch.cat(parts, dim=1)

    def mixed_loss(
        self, xb: torch.Tensor, out: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (loss_multi, loss_gauss) for one noised batch.

        loss_gauss: MSE of the predicted noise on numerical columns.
        loss_multi: cross-entropy of the predicted clean category per column.
        """
        y = out.get("y") if isinstance(out, dict) else None
        x0 = self._to_internal(xb)
        b = x0.shape[0]
        t = torch.randint(0, self._steps, (b,), device=xb.device)
        ab = self._alpha_bar[t].view(-1, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = self._denoise_fn(x_t, t, y)

        loss_gauss = torch.tensor(0.0, device=xb.device)
        loss_multi = torch.tensor(0.0, device=xb.device)
        if self._n_num > 0:
            loss_gauss = F.mse_loss(pred[:, : self._n_num], eps[:, : self._n_num])
        if self._n_cat > 0:
            ce = [
                F.cross_entropy(
                    pred[:, sl], xb[:, self._n_num + j].long().clamp(0, k - 1)
                )
                for j, (sl, k) in enumerate(zip(self._cat_slices(), self._cat_sizes))
            ]
            loss_multi = torch.stack(ce).mean()
        return loss_multi, loss_gauss

    @torch.no_grad()
    def sample_all(
        self,
        num_samples: int,
        *,
        batch_size: int,
        y_dist: torch.Tensor,
        ddim: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic x and y by ancestral DDPM sampling.

        y is drawn from ``y_dist`` and the denoiser is conditioned on it.
        Numerical columns end as the final mean; each categorical column is the
        argmax of its predicted clean block.
        """
        device = self._dummy.device
        n = int(num_samples)

        if y_dist is not None and y_dist.numel() > 1:
            y_dist = (y_dist / y_dist.sum()).to(device)
            y_vec = torch.multinomial(y_dist, n, replacement=True)
        else:
            y_vec = torch.zeros(n, dtype=torch.long, device=device)

        d_internal = self._n_num + sum(self._cat_sizes)
        chunk = max(int(batch_size), self.SAMPLE_BATCH)
        outputs = []
        for start in range(0, n, chunk):
            yb = y_vec[start : start + chunk]
            bsz = yb.shape[0]
            x = torch.randn(bsz, d_internal, device=device)
            last_pred = None
            for step in reversed(range(self._steps)):
                t = torch.full((bsz,), step, device=device, dtype=torch.long)
                pred = self._denoise_fn(x, t, yb)
                last_pred = pred
                ab = self._alpha_bar[step]
                alpha = self._alphas[step]
                beta = self._betas[step]
                eps = pred.clone()
                for sl in self._cat_slices():
                    x0_hat = torch.softmax(pred[:, sl], dim=1) * 2.0 - 1.0
                    eps[:, sl] = (x[:, sl] - ab.sqrt() * x0_hat) / (1 - ab).sqrt()
                mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
                x = mean + beta.sqrt() * torch.randn_like(x) if step > 0 else mean
            cols = []
            if self._n_num > 0:
                cols.append(x[:, : self._n_num].clamp(-3.0, 3.0))
            for sl in self._cat_slices():
                cols.append(last_pred[:, sl].argmax(dim=1, keepdim=True).float())
            outputs.append(torch.cat(cols, dim=1) if cols else x)

        return torch.cat(outputs, dim=0)[:n], y_vec[:n]
