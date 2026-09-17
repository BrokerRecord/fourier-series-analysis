"""Fourier series coefficient calculation and partial sum reconstruction."""

from typing import Callable, Tuple
import numpy as np
from scipy.integrate import quad


class FourierSeries:
    """Computes Fourier coefficients and evaluates partial sum reconstructions.

    Standard real Fourier series on [-L, L]:
        f(x) ~ a0 / 2 + sum_{n=1}^N [ an * cos(n pi x / L) + bn * sin(n pi x / L) ]
    """

    def __init__(self, func: Callable[[float], float], L: float = np.pi):
        self.func = func
        self.L = L

    def compute_coefficients(self, n_modes: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute Fourier coefficients (a0, an, bn) up to `n_modes` using Gauss-Kronrod quadrature."""
        # a0 = (1 / L) * integral_{-L}^L f(x) dx
        a0, _ = quad(self.func, -self.L, self.L, limit=200)
        a0 /= self.L

        an = np.zeros(n_modes)
        bn = np.zeros(n_modes)

        for n in range(1, n_modes + 1):
            an_val, _ = quad(
                lambda x: self.func(x) * np.cos(n * np.pi * x / self.L),
                -self.L,
                self.L,
                limit=200,
            )
            bn_val, _ = quad(
                lambda x: self.func(x) * np.sin(n * np.pi * x / self.L),
                -self.L,
                self.L,
                limit=200,
            )
            an[n - 1] = an_val / self.L
            bn[n - 1] = bn_val / self.L

        return a0, an, bn

    def reconstruct(self, x: np.ndarray, n_modes: int) -> np.ndarray:
        """Evaluate the N-mode Fourier partial sum on spatial array `x`."""
        a0, an, bn = self.compute_coefficients(n_modes)

        # Base DC component
        y = np.full_like(x, a0 / 2.0, dtype=np.float64)

        # Vectorized mode evaluation
        n_vector = np.arange(1, n_modes + 1).reshape(-1, 1)  # Shape (N, 1)
        x_vector = x.reshape(1, -1)                          # Shape (1, M)

        arg = (n_vector * np.pi * x_vector) / self.L

        cos_terms = np.cos(arg)  # Shape (N, M)
        sin_terms = np.sin(arg)  # Shape (N, M)

        y += np.dot(an, cos_terms) + np.dot(bn, sin_terms)
        return y