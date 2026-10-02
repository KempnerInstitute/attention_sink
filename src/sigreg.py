"""
SIGReg: Sketched Isotropic Gaussian Regularization.

From LeJEPA (Balestriero & LeCun, 2025). Self-contained implementation based on
the minimal working example — no external `lejepa` pip dependency needed.

The loss constrains learned embeddings to follow an isotropic Gaussian distribution
by comparing the empirical characteristic function (via random slicing) against the
target Gaussian characteristic function using numerical quadrature.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss.

    Args:
        knots: Number of quadrature points for the ECF comparison (default 17).
        t_max: Upper integration limit (default 3.0).
        num_slices: Number of random projection directions (default 256).
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices

        # Precompute quadrature nodes, weights, and target Gaussian CF values.
        # We integrate on [0, t_max] and exploit symmetry (double weights) for efficiency.
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal rule endpoint correction
        window = torch.exp(-t.square() / 2.0)  # target Gaussian CF: exp(-t^2/2)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings of shape (V, N, proj_dim) or (N, proj_dim).
                  If 3D, views are treated as independent samples for the statistic.

        Returns:
            Scalar loss value.
        """
        if proj.dim() == 3:
            # Flatten views into the sample dimension: (V, N, D) -> (V*N, D)
            proj = proj.reshape(-1, proj.size(-1))

        # Random slicing directions (unit vectors)
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # Project onto random directions and compute ECF at quadrature points
        # x_t shape: (N, num_slices, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Empirical CF vs target Gaussian CF
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()

        # Weighted quadrature integral, scaled by sample size
        statistic = (err @ self.weights) * proj.size(0)

        return statistic.mean()
