"""
Tests for mach3sbitools.simulator.priors.cyclical_distribution.

Statistical property tests share a single large sample draw via a
session-scoped fixture to avoid redundant sampling overhead.
"""

from typing import cast

import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from mach3sbitools.simulator.priors.cyclical_distribution import CyclicalDistribution


@pytest.fixture(scope="session")
def cyclical_distribution() -> CyclicalDistribution:
    return CyclicalDistribution(torch.ones(1))


@pytest.fixture(scope="session")
def large_samples(cyclical_distribution) -> torch.Tensor:
    """50k samples shared across all statistical property tests."""
    return cast(torch.Tensor, cyclical_distribution.sample(torch.Size([50_000])))


# ─────────────────────────────────────────────────────────────────────────────
# Analytical properties
# ─────────────────────────────────────────────────────────────────────────────


def test_log_prob(cyclical_distribution):
    theta = torch.tensor([[0.0], [10.0], [1.0]], dtype=torch.double)
    l_one = 0.5 * (np.sin(0.25 * (1 + 2 * np.pi)) ** 2) / np.pi
    expected = torch.tensor(
        [[np.log(0.5 / np.pi)], [torch.tensor(np.float64("-inf"))], [np.log(l_one)]],
        dtype=torch.double,
    )
    assert torch.allclose(cyclical_distribution.log_prob(theta), expected)


def test_cdf(cyclical_distribution):
    theta = torch.tensor([[0.0], [10.0], [1.0]], dtype=torch.double)
    cdf_one = 0.5 * (0.5 + np.sin(0.5) + np.pi) / np.pi
    expected = torch.tensor([[0.5], [1.0], [cdf_one]], dtype=torch.double)
    assert torch.allclose(cyclical_distribution.cdf(theta), expected)


# ─────────────────────────────────────────────────────────────────────────────
# Sample shape — parametrized
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,expected",
    [
        (torch.Size([100]), torch.Size([100, 1])),
        (torch.Size([]), torch.Size([1])),
    ],
)
def test_sample_shape(cyclical_distribution, shape, expected):
    assert cyclical_distribution.sample(shape).shape == expected


# ─────────────────────────────────────────────────────────────────────────────
# Statistical properties — shared large_samples fixture
# ─────────────────────────────────────────────────────────────────────────────


def test_sample_in_bounds(large_samples):
    assert torch.all(large_samples >= -2 * torch.pi)
    assert torch.all(large_samples <= 2 * torch.pi)


def test_sample_mean_near_zero(large_samples):
    """Distribution is symmetric around 0 so mean should be close to 0."""
    assert torch.abs(large_samples.mean()) < 0.1


def test_sample_cdf_uniformity(cyclical_distribution, large_samples):
    """CDF(samples) ~ Uniform(0,1) by probability integral transform."""
    u = cyclical_distribution.cdf(large_samples)
    assert torch.abs(u.mean() - 0.5) < 0.05
    assert torch.abs(u.std() - (1 / 12) ** 0.5) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Two-sample KS test against accept-reject reference
# ─────────────────────────────────────────────────────────────────────────────


def test_against_mc(cyclical_distribution, large_samples):
    """
    Compare samples against an accept-reject Monte Carlo reference.
    Uses CLT-derived tolerances so the test scales correctly with n.
    """
    rng = np.random.default_rng(42)
    n = 50_000
    lower, upper = -2 * np.pi, 2 * np.pi
    M = 0.5 / np.pi  # uniform envelope = max of PDF

    accepted: list[float] = []
    while len(accepted) < n:
        batch = 2 * n
        x = rng.uniform(lower, upper, size=batch)
        y = rng.uniform(0, M, size=batch)
        pdf_vals = (
            cyclical_distribution.pdf(torch.tensor(x, dtype=torch.double).unsqueeze(-1))
            .squeeze()
            .numpy()
        )
        accepted.extend(x[y < pdf_vals].tolist())

    ref = np.array(accepted[:n])
    samples = large_samples.squeeze().numpy()

    mean_tol = 5 * np.sqrt((samples.var() + ref.var()) / n)
    assert abs(samples.mean() - ref.mean()) < mean_tol

    var_tol = 5 * samples.var() ** 2 * np.sqrt(2 / (n - 1))
    assert abs(samples.var() - ref.var()) < var_tol

    _, p = ks_2samp(samples, ref)
    assert p > 0.05, f"KS test failed: p={p:.4f}"
