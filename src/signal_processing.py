"""Comparative analysis between analytical Fourier Series and Fast Fourier Transform (FFT)."""

from typing import Dict
import numpy as np
from src.fourier import FourierSeries


def compare_fourier_vs_fft(
    func: callable, L: float = np.pi, n_samples: int = 1024, n_modes: int = 32
) -> Dict[str, np.ndarray]:
    """Compares spatial series reconstruction with DFT/FFT spectrum coefficients."""
    x = np.linspace(-L, L, n_samples, endpoint=False)
    f_x = np.vectorize(func)(x)

    # 1. Continuous Fourier Series Reconstruction
    fs = FourierSeries(func, L=L)
    y_fourier = fs.reconstruct(x, n_modes=n_modes)

    # 2. Discrete Fourier Transform via FFT
    fft_coeffs = np.fft.rfft(f_x) / n_samples
    freqs = np.fft.rfftfreq(n_samples, d=(2 * L) / n_samples)

    # Reconstruct signal from top N FFT modes
    fft_truncated = np.zeros_like(fft_coeffs)
    fft_truncated[: n_modes + 1] = fft_coeffs[: n_modes + 1]
    y_fft = np.fft.irfft(fft_truncated * n_samples, n=n_samples)

    return {
        "x": x,
        "exact": f_x,
        "fourier_series": y_fourier,
        "fft_reconstruction": y_fft,
        "freqs": freqs[: n_modes + 1],
        "fft_amplitudes": np.abs(fft_coeffs[: n_modes + 1]),
    }