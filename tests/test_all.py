"""Unit tests for Fourier coefficients, convergence, and Gibbs calculation."""

import numpy as np
import pytest

from src.fourier import FourierSeries
from src.gibbs_analysis import (
    analyze_gibbs_overshoot,
    square_wave,
    theoretical_gibbs_constant,
    theoretical_gibbs_fraction,
)
from src.signal_processing import compare_fourier_vs_fft


# ---------------------------------------------------------------------------
# Fourier coefficients
# ---------------------------------------------------------------------------
def test_sine_wave_coefficients():
    """Single mode input sin(x) → a1 = 0, b1 = 1, all higher modes zero."""
    func = lambda x: np.sin(x)
    fs = FourierSeries(func, L=np.pi)
    a0, an, bn = fs.compute_coefficients(n_modes=5)

    assert np.isclose(a0, 0.0, atol=1e-10)
    assert np.allclose(an, 0.0, atol=1e-10)
    assert np.isclose(bn[0], 1.0, atol=1e-10)
    assert np.allclose(bn[1:], 0.0, atol=1e-10)


def test_square_wave_odd_symmetry():
    """Odd square wave → a_n = 0 everywhere; b_n = 4/(nπ) for odd n, 0 else."""
    fs = FourierSeries(square_wave, L=np.pi)
    a0, an, bn = fs.compute_coefficients(n_modes=6)

    assert np.isclose(a0, 0.0, atol=1e-8)
    assert np.allclose(an, 0.0, atol=1e-8)

    # odd modes: b_1 = 4/π, b_3 = 4/(3π), b_5 = 4/(5π)
    assert np.isclose(bn[0], 4.0 / (1.0 * np.pi), atol=1e-6)
    assert np.isclose(bn[2], 4.0 / (3.0 * np.pi), atol=1e-6)
    assert np.isclose(bn[4], 4.0 / (5.0 * np.pi), atol=1e-6)

    # even modes vanish
    assert np.isclose(bn[1], 0.0, atol=1e-6)
    assert np.isclose(bn[3], 0.0, atol=1e-6)
    assert np.isclose(bn[5], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Gibbs constants
# ---------------------------------------------------------------------------
def test_gibbs_theoretical_constant_absolute():
    """Absolute overshoot for a unit-amplitude square wave: ≈ 0.178980."""
    assert np.isclose(theoretical_gibbs_constant(), 0.178980, atol=1e-6)


def test_gibbs_theoretical_constant_fractional():
    """Classic fractional overshoot ('8.95 %'): ≈ 0.0894898."""
    assert np.isclose(theoretical_gibbs_fraction(), 0.0894898, atol=1e-6)


def test_gibbs_constants_consistency():
    """Fractional = absolute / 2, since the jump size is 2."""
    assert np.isclose(
        theoretical_gibbs_constant() / 2.0,
        theoretical_gibbs_fraction(),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Gibbs overshoot convergence
# ---------------------------------------------------------------------------
def test_gibbs_overshoot_converges_to_limit():
    """Measured overshoot must approach the absolute theoretical limit."""
    modes = [10, 50, 100, 500, 1000]
    res = analyze_gibbs_overshoot(modes, samples_per_mode=400)

    # Basic shape
    assert res["modes"].shape == (len(modes),)
    assert res["overshoots"].shape == (len(modes),)

    # Every measured value should be close to 0.17898 — the whole point
    # of the Gibbs phenomenon is that the overshoot does *not* die out.
    assert np.allclose(res["overshoots"], 0.178980, atol=5e-3)

    # And the last one should be tight
    assert np.isclose(res["overshoots"][-1], res["theoretical_limit"], atol=5e-3)


def test_gibbs_overshoot_monotone_approach():
    """Overshoot should converge monotonically from below (in absolute value)."""
    res = analyze_gibbs_overshoot([10, 50, 200, 1000], samples_per_mode=400)
    diffs = np.abs(res["overshoots"] - res["theoretical_limit"])

    # Not strictly monotone for every pair, but the overall trend should
    # be decreasing — check that the last gap is smaller than the first.
    assert diffs[-1] < diffs[0]
    
# ---------------------------------------------------------------------------
# Fourier series vs FFT
# ---------------------------------------------------------------------------



def _smooth_signal(x):
    return np.cos(x) + 0.3 * np.cos(3 * x) - 0.5 * np.sin(2 * x)


def test_compare_fourier_vs_fft_smooth_agreement():
    """For a smooth signal, FS and FFT reconstructions should agree to ~1e-12."""
    out = compare_fourier_vs_fft(_smooth_signal, L=np.pi, n_samples=1024, n_modes=16)

    # basic shapes
    assert out["x"].shape == (1024,)
    assert out["exact"].shape == (1024,)
    assert out["fourier_series"].shape == (1024,)
    assert out["fft_reconstruction"].shape == (1024,)

    # smooth signal: machine-precision agreement
    err = np.max(np.abs(out["fourier_series"] - out["fft_reconstruction"]))
    assert err < 1e-10


def test_compare_fourier_vs_fft_returned_keys():
    """The dict should expose exactly the keys the notebooks rely on."""
    out = compare_fourier_vs_fft(_smooth_signal, L=np.pi, n_samples=256, n_modes=8)

    expected_keys = {
        "x",
        "exact",
        "fourier_series",
        "fft_reconstruction",
        "freqs",
        "fft_amplitudes",
    }
    assert set(out.keys()) == expected_keys
    assert out["freqs"].shape == (9,)         # n_modes + 1
    assert out["fft_amplitudes"].shape == (9,)


def test_compare_fourier_vs_fft_aliasing_decreases():
    """For a jump signal, the FS–FFT discrepancy shrinks as n_samples grows."""
    errs = []
    for n in (64, 256, 1024):
        out = compare_fourier_vs_fft(square_wave, L=np.pi, n_samples=n, n_modes=16)
        errs.append(np.max(np.abs(out["fft_reconstruction"] - out["fourier_series"])))

    # The last sampling density should beat the first
    assert errs[-1] < errs[0]