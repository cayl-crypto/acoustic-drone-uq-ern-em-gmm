from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialNIGLossWithKL(nn.Module):
    """
    Evidential NIG loss + KL divergence to a prior NIG(mu0, v0, alpha0, beta0).
    Inputs are already activated (v>0, alpha>1, beta>0).
    """

    def __init__(
        self,
        coeff: float = 1e-1,
        kl_weight: float = 1e-1,
        prior_mu: float = 0.0,
        prior_v: float = 1.0,
        prior_alpha: float = 1.5,
        prior_beta: float = 1.0,
    ):
        super().__init__()
        self.coeff = coeff
        self.kl_weight = kl_weight
        self.mu0 = prior_mu
        self.v0 = prior_v
        self.alpha0 = prior_alpha
        self.beta0 = prior_beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mu, v, alpha, beta = torch.chunk(y_pred, 4, dim=-1)

        eps = 1e-6
        v = torch.clamp(v, min=eps)
        alpha = torch.clamp(alpha, min=1.0 + eps)
        beta = torch.clamp(beta, min=eps)
        # v = F.softplus(v) + 1e-6
        # alpha = F.softplus(alpha) + 1.0 + 1e-6
        # beta = F.softplus(beta) + 1e-6

        y_true = y_true.view_as(mu)

        two_blambda = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(two_blambda)
            + (alpha + 0.5) * torch.log(v * (y_true - mu) ** 2 + two_blambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        err = torch.abs(y_true - mu).detach()
        evidence = 2.0 * v + alpha  # evidence (must NOT be detached)
        reg = err * evidence  # or (err**2) * evidence
        # reg = torch.abs((y_true - mu).detach())

        kl = self.kl_divergence(mu, v, alpha, beta)

        return nll.mean() + self.coeff * reg.mean() + self.kl_weight * kl.mean()

    def kl_divergence(self, mu, v, alpha, beta) -> torch.Tensor:
        """
        KL( NIG(mu,v,alpha,beta) || NIG(mu0,v0,alpha0,beta0) )
        Uses tensor priors (broadcast-safe) on the correct device/dtype.
        """
        mu0 = torch.as_tensor(self.mu0, device=mu.device, dtype=mu.dtype)
        v0 = torch.as_tensor(self.v0, device=v.device, dtype=v.dtype)
        alpha0 = torch.as_tensor(self.alpha0, device=alpha.device, dtype=alpha.dtype)
        beta0 = torch.as_tensor(self.beta0, device=beta.device, dtype=beta.dtype)

        term1 = 0.5 * (
            torch.log(v0 / v)
            + v / v0
            + (v0 * (mu - mu0) ** 2) / (beta0 / (alpha0 - 1.0))
            - 1.0
        )

        term2 = (
            alpha0 * torch.log(beta / beta0)
            - torch.lgamma(alpha)
            + torch.lgamma(alpha0)
            + (alpha - alpha0) * torch.digamma(alpha)
            - (beta - beta0) * (alpha / beta)
        )

        return term1 + term2
