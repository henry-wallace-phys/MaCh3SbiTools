import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp  # ← correct two-sample function

from mach3sbitools.simulator.priors.cyclical_distribution import CyclicalDistribution


@pytest.fixture(scope="session")
def cyclical_distribution() -> CyclicalDistribution:
    nominals = torch.ones(1)
    lower_bounds = -2 * torch.pi * torch.ones(1)
    upper_bounds = 2 * torch.pi * torch.ones(1)
    return CyclicalDistribution(nominals, lower_bounds, upper_bounds)


def test_cyclical_bound_error():
    nominals = torch.tensor([1, 1])
    lower_bounds = torch.tensor([0, -2 * torch.pi])
    upper_bounds = 2 * torch.pi * torch.ones(2)
    # Check lower bounds wrong
    with pytest.raises(NotImplementedError):
        CyclicalDistribution(nominals, lower_bounds, upper_bounds)

    # Check upper bounds wrong
    lower_bounds = -2 * torch.pi * torch.ones(2)
    upper_bounds = torch.tensor([1, 2 * torch.pi])
    with pytest.raises(NotImplementedError):
        CyclicalDistribution(nominals, lower_bounds, upper_bounds)


def test_log_prob(cyclical_distribution):
    theta_values = torch.tensor([[0.0], [10.0], [1.0]], dtype=torch.double)

    # Need to calculate the likelihood
    l_one = 0.5 * (np.sin(0.25 * (1 + 2 * np.pi)) ** 2) / np.pi
    llh_one = np.log(l_one)
    expected_log_pdf = torch.tensor(
        [[np.log(0.5 / np.pi)], [torch.tensor(np.float64("-inf"))], [llh_one]],
        dtype=torch.double,
    )

    likelihoods = cyclical_distribution.log_prob(theta_values)
    assert torch.allclose(likelihoods, expected_log_pdf)


def test_cdf(cyclical_distribution):
    theta_values = torch.tensor([[0.0], [10.0], [1.0]], dtype=torch.double)

    cdf_one = 0.5 * (0.5 + np.sin(0.5) + np.pi) / np.pi
    expected_cdf = torch.tensor([[1 / 2], [1], [cdf_one]], dtype=torch.double)
    cdf = cyclical_distribution.cdf(theta_values)

    assert torch.allclose(cdf, expected_cdf)


def test_sample_shape(cyclical_distribution):
    """Samples have the correct shape."""
    samples = cyclical_distribution.sample(torch.Size([100]))
    assert samples.shape == torch.Size([100, 1])


def test_sample_shape_scalar(cyclical_distribution):
    """Calling sample() with no arguments returns a single event."""
    samples = cyclical_distribution.sample()
    assert samples.shape == torch.Size([1])


def test_sample_in_bounds(cyclical_distribution):
    """All samples fall within [-2pi, 2pi]."""
    samples = cyclical_distribution.sample(torch.Size([1000]))
    assert torch.all(samples >= -2 * torch.pi)
    assert torch.all(samples <= 2 * torch.pi)


def test_sample_mean(cyclical_distribution):
    """
    Sample mean should be close to 0 by symmetry of the distribution
    around 0 on [-2pi, 2pi].
    """
    samples = cyclical_distribution.sample(torch.Size([10_000]))
    assert torch.abs(samples.mean()) < 0.1


def test_sample_cdf_uniformity(cyclical_distribution):
    """
    Applying the CDF to samples should give approximately Uniform(0,1).
    This is the probability integral transform test - if sampling is correct,
    CDF(samples) ~ Uniform(0, 1) with mean ~0.5 and std ~1/sqrt(12).
    """
    samples = cyclical_distribution.sample(torch.Size([50_000]))
    u = cyclical_distribution.cdf(samples)

    assert torch.abs(u.mean() - 0.5) < 0.05
    assert torch.abs(u.std() - (1 / 12) ** 0.5) < 0.05


def test_against_mc(cyclical_distribution):
    """
    Compare samples from CyclicalDistribution against an accept-reject
    Monte Carlo reference using the true PDF.

    Robustness improvements vs. the naive version:
      - Fixed RNG seeds make the test deterministic.
      - Mean / variance tolerances are derived from the CLT (5-sigma), so
        they scale correctly with n and the distribution's own spread.
      - ks_2samp is used explicitly for the two-sample KS test.
    """
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    n_samples = 50_000
    lower, upper = -2 * np.pi, 2 * np.pi
    M = 0.5 / np.pi  # uniform envelope (= max of the PDF)

    # --- Build reference via accept-reject ---
    accepted: list[float] = []
    while len(accepted) < n_samples:
        batch = 2 * n_samples
        x = rng.uniform(lower, upper, size=batch)
        y = rng.uniform(0, M, size=batch)
        pdf_vals = (
            cyclical_distribution.pdf(torch.tensor(x, dtype=torch.double).unsqueeze(-1))
            .squeeze()
            .numpy()
        )
        accepted.extend(x[y < pdf_vals].tolist())

    ref = np.array(accepted[:n_samples])
    samples = cyclical_distribution.sample(torch.Size([n_samples])).squeeze().numpy()

    # --- CLT-derived mean tolerance (5-sigma) ---
    # Under H₀: (mean_s - mean_r) ~ N(0, (σ_s² + σ_r²) / n)
    mean_tol = 5 * np.sqrt((samples.var() + ref.var()) / n_samples)
    assert abs(samples.mean() - ref.mean()) < mean_tol, (
        f"Means differ by {abs(samples.mean() - ref.mean()):.4f}, "
        f"tolerance {mean_tol:.4f}"
    )

    # --- CLT-derived variance tolerance (5-sigma) ---
    # Var of sample variance estimator ~ 2σ⁴ / (n-1)
    var_tol = 5 * samples.var() ** 2 * np.sqrt(2 / (n_samples - 1))
    assert abs(samples.var() - ref.var()) < var_tol, (
        f"Variances differ by {abs(samples.var() - ref.var()):.4f}, "
        f"tolerance {var_tol:.4f}"
    )

    # --- Two-sample KS test ---
    ks_stat, p_value = ks_2samp(samples, ref)
    assert p_value > 0.05, f"KS test failed: stat={ks_stat:.4f}, p={p_value:.4f}"
