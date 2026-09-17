"""Unit tests for Fourier coefficients, convergence, and Gibbs calculation."""

import numpy as np
import pytest
from src.fourier import FourierSeries
from src.gibbs_analysis import square_wave, theoretical_gibbs_constant, analyze_gibbs_overshoot


def test_sine_wave_coefficients():
    """Single mode input sin(x) should yield a1=0, b1=1, and zero for all higher modes."""
    func = lambda x: np.sin(x)
    fs = FourierSeries(func, L=np.pi)
    a0, an, bn = fs.compute_coefficients(n_modes=5)

    assert np.isclose(a0, 0.0, atol=1e-10)
    assert np.allclose(an, 0.0, atol=1e-10)
    assert np.isclose(bn[0], 1.0, atol=1e-10)
    assert np.allclose(bn[1:], 0.0, atol=1e-10)


def test_square_wave_odd_symmetry():
    """Odd square wave should yield an = 0 everywhere (only sine terms bn)."""
    fs = FourierSeries(square_wave, L=np.pi)
    a0, an, bn = fs.compute_coefficients(n_modes=4)

    assert np.isclose(a0, 0.0, atol=1e-8)
    assert np.allclose(an, 0.0, atol=1e-8)
    # Odd modes b1, b3 should match theoretical 4/(n*pi)
    assert np.isclose(bn[0], 4.0 / np.pi, atol=1e-3)
    assert np.isclose(bn[2], 4.0 / (3.0 * np.pi), atol=1e-3)


def test_gibbs_theoretical_limit():
    """Verify Wilbraham-Gibbs constant evaluates to ~8.95%."""
    limit = theoretical_gibbs_constant()
    assert np.isclose(limit, 0.0894898, atol=1e-5)


def test_gibbs_overshoot_convergence():
    """Ensure computed overshoot converges near the theoretical limit as modes increase."""
    res = analyze_gibbs_overshoot(n_modes_list=[10, 50], resolution=5000)
    last_overshoot = res["overshoots"][-1]
    assert np.isclose(last_overshoot, res["theoretical_limit"], atol=0.02)