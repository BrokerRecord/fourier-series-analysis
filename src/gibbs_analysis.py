"""Quantitative study of the Gibbs Phenomenon near jump discontinuities."""

from typing import Dict, List

import numpy as np
from scipy.special import sici

from src.fourier import FourierSeries


# ---------------------------------------------------------------------------
# Test signal
# ---------------------------------------------------------------------------
def square_wave(x: float) -> float:
    """Unit-amplitude square wave on [-pi, pi] with a jump at x = 0."""
    return 1.0 if x >= 0 else -1.0


# ---------------------------------------------------------------------------
# Theoretical constants
# ---------------------------------------------------------------------------
def theoretical_gibbs_constant() -> float:
    
    si_pi, _ = sici(np.pi)
    return float((2.0 / np.pi) * si_pi - 1.0)


def theoretical_gibbs_fraction() -> float:

    si_pi, _ = sici(np.pi)
    return float(si_pi / np.pi - 0.5)


# ---------------------------------------------------------------------------
# Analytic square-wave partial sum
# ---------------------------------------------------------------------------
def square_wave_partial_sum(x: np.ndarray, n_modes: int) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    # Only odd harmonics contribute.
    for n in range(1, n_modes + 1, 2):
        y += (4.0 / (n * np.pi)) * np.sin(n * x)
    return y


# ---------------------------------------------------------------------------
# Numerical Gibbs study
# ---------------------------------------------------------------------------
def analyze_gibbs_overshoot(
    n_modes_list: List[int],
    samples_per_mode: int = 400,
) -> Dict[str, np.ndarray]:
    
    overshoots = np.empty(len(n_modes_list), dtype=np.float64)

    for i, N in enumerate(n_modes_list):

        x = np.linspace(0.0, np.pi / N, samples_per_mode)
        y = square_wave_partial_sum(x, n_modes=N)
        overshoots[i] = float(np.max(y) - 1.0)

    return {
        "modes": np.asarray(n_modes_list, dtype=int),
        "overshoots": overshoots,
        "theoretical_limit": theoretical_gibbs_constant(),
    }