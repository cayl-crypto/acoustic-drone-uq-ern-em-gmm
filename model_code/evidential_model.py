# model_evidential.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from evidential_loss import EvidentialNIGLossWithKL  # <-- make sure path is correct


class EvidentialLocalization(nn.Module):
    """
    Evidential head for localization only (no classifier).
    Three NIG heads: sin, cos, range.

    forward(x)            -> (sin_mu, cos_mu, range_mu)           # means only
    forward_params(x)     -> ((sin_mu, lam, alpha, beta), ...)    # tuples per head
    forward_all(x)        -> (means_tuple, packs_tuple)           # means + (B,4) packs
    compute_loss(params, targets)                                 # evidential loss (sum of three heads)

    Notes:
    - This head applies the necessary activations:
        * sin/cos mu via tanh,  range mu via sigmoid
        * lam via softplus + eps, alpha via softplus + 1 + eps, beta via softplus + eps
    - The loss functions should NOT softplus again (they should only clamp).
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        coeff: float = 0.1,
        kl_weight: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.eps = eps

        self.sin_head = nn.Linear(
            embed_dim, 4
        )  # (mu_raw, log_lambda, log_alpha, log_beta)
        self.cos_head = nn.Linear(embed_dim, 4)
        self.range_head = nn.Linear(embed_dim, 4)

        self.softplus = nn.Softplus()
        self.sin_evidential_loss = EvidentialNIGLossWithKL(
            coeff=coeff, kl_weight=kl_weight
        )
        self.cos_evidential_loss = EvidentialNIGLossWithKL(
            coeff=coeff, kl_weight=kl_weight
        )
        self.range_evidential_loss = EvidentialNIGLossWithKL(
            coeff=coeff, kl_weight=kl_weight
        )

    def _decode(self, four_vec: torch.Tensor, is_range: bool):
        """
        Input: four_vec (B, 4) with raw outputs.
        Returns: (mu, lam, alpha, beta) each (B,)
        """
        mu_raw, log_lambda, log_alpha, log_beta = four_vec.chunk(4, dim=-1)

        # squash means to valid ranges
        mu = torch.sigmoid(mu_raw) if is_range else torch.tanh(mu_raw)

        # activate NIG params (already-positive for the loss)
        lam = self.softplus(log_lambda) + self.eps
        alpha = self.softplus(log_alpha) + 1.0 + self.eps
        beta = self.softplus(log_beta) + self.eps

        # squeeze last dim -> (B,)
        return mu.squeeze(-1), lam.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)

    def forward_params(self, x: torch.Tensor):
        """Return per-head tuples (mu, lam, alpha, beta), each (B,)."""
        z = self.encoder(x)  # (B, D)
        sin_p = self._decode(self.sin_head(z), is_range=False)
        cos_p = self._decode(self.cos_head(z), is_range=False)
        range_p = self._decode(self.range_head(z), is_range=True)
        return sin_p, cos_p, range_p

    def forward(self, x: torch.Tensor):
        """Means only, for metric/evaluator compatibility."""
        sin_p, cos_p, range_p = self.forward_params(x)
        return sin_p[0], cos_p[0], range_p[0]  # (sin_mu, cos_mu, range_mu)

    def forward_all(self, x: torch.Tensor):
        """
        Convenience: returns ((sin_mu, cos_mu, range_mu),
                              (sin_pack, cos_pack, range_pack)), where each pack is (B,4).
        """
        sin_p, cos_p, range_p = self.forward_params(x)
        sin_mu, sin_lam, sin_alpha, sin_beta = sin_p
        cos_mu, cos_lam, cos_alpha, cos_beta = cos_p
        rng_mu, rng_lam, rng_alpha, rng_beta = range_p

        sin_pack = torch.stack((sin_mu, sin_lam, sin_alpha, sin_beta), dim=-1)  # (B,4)
        cos_pack = torch.stack((cos_mu, cos_lam, cos_alpha, cos_beta), dim=-1)
        range_pack = torch.stack((rng_mu, rng_lam, rng_alpha, rng_beta), dim=-1)

        means = (sin_mu, cos_mu, rng_mu)
        packs = (sin_pack, cos_pack, range_pack)
        return means, packs

    def compute_loss(self, params, targets):
        """
        params: ((sin_mu, sin_lam, sin_alpha, sin_beta),
                 (cos_mu, cos_lam, cos_alpha, cos_beta),
                 (rng_mu, rng_lam, rng_alpha, rng_beta))
        targets: (y_sin, y_cos, y_range) in the same dtype/device as the outputs.
        """
        (sin_p, cos_p, range_p) = params
        (y_sin, y_cos, y_range) = targets

        # pack as (B,4) for each head
        sin_pack = torch.stack(sin_p, dim=-1)
        cos_pack = torch.stack(cos_p, dim=-1)
        range_pack = torch.stack(range_p, dim=-1)

        # align target dtypes/devices and add channel dim
        y_sin = y_sin.to(sin_pack.dtype).unsqueeze(-1)
        y_cos = y_cos.to(cos_pack.dtype).unsqueeze(-1)
        y_range = y_range.to(range_pack.dtype).unsqueeze(-1)

        loss = (
            self.sin_evidential_loss(sin_pack, y_sin)
            + self.cos_evidential_loss(cos_pack, y_cos)
            + self.range_evidential_loss(range_pack, y_range)
        )
        return loss

    @staticmethod
    def compute_metrics_from_means(
        sin_mu: torch.Tensor,
        cos_mu: torch.Tensor,
        range_mu: torch.Tensor,
        y_sin: torch.Tensor,
        y_cos: torch.Tensor,
        y_range: torch.Tensor,
        max_range: float = 250.0,
    ) -> dict:
        """Bearing MAE (deg), Range MAE (m), Euclidean MAE (m) from means."""
        # bearing error (wrap to (-pi, pi])
        pred_bearing = torch.atan2(sin_mu, cos_mu)
        true_bearing = torch.atan2(y_sin, y_cos)
        d = (pred_bearing - true_bearing + math.pi) % (2 * math.pi) - math.pi
        bearing_mae_deg = torch.rad2deg(d.abs()).mean().item()

        # range error (meters)
        r_hat_m = range_mu * max_range
        r_true_m = y_range * max_range
        range_mae_m = torch.mean(torch.abs(r_hat_m - r_true_m)).item()

        # euclidean error (meters)
        px = r_hat_m * cos_mu
        py = r_hat_m * sin_mu
        tx = r_true_m * y_cos
        ty = r_true_m * y_sin
        euc_mae_m = (
            torch.norm(torch.stack([px - tx, py - ty], dim=-1), dim=-1).mean().item()
        )

        return {
            "bearing_mae_deg": float(bearing_mae_deg),
            "range_mae_m": float(range_mae_m),
            "euclidean_mae_m": float(euc_mae_m),
        }
