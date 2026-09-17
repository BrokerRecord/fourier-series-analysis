"""Quantitative study of the Gibbs Phenomenon near jump discontinuities."""

from typing import Dict, Tuple
import numpy as np
from scipy.special import sici
from src.fourier import FourierSeries


def square_wave(x: float) -> float:
    """Standard square wave step function on [-pi, pi]."""
    return 1.0 if x >= 0 else -1.0


def theoretical_gibbs_constant() -> float:
    """Calculates the analytical Wilbraham-Gibbs constant limit:

    (2 / pi) * Si(pi) - 1 ≈ 0.0894898 (8.95% relative overshoot)
    """
    si_pi, _ = sici(np.pi)
    return float((2.0 / np.pi) * si_pi - 1.0)


def analyze_gibbs_overshoot(n_modes_list: list[int], resolution: int = 10000) -> Dict[str, np.ndarray]:
    """Measures the maximum overshoot near x = 0 for a square wave across modes."""
    fs = FourierSeries(square_wave, L=np.pi)
    x = np.linspace(-np.pi, np.pi, resolution)

    overshoots = []
    theoretical_limit = theoretical_gibbs_constant()

    for N in n_modes_list:
        y_approx = fs.reconstruct(x, n_modes=N)
        # Find peak value in right-hand neighborhood of jump discontinuity
        right_half_mask = (x > 0) & (x < np.pi / 2)
        peak_val = np.max(y_approx[right_half_mask])

        # Relative overshoot ratio above the step size height of 1.0
        overshoot = peak_val - 1.0
        overshoots.append(overshoot)

    return {
        "modes": np.array(n_modes_list),
        "overshoots": np.array(overshoots),
        "theoretical_limit": theoretical_limit,
    }