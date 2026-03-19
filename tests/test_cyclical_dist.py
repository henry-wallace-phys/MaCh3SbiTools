import numpy as np
import pytest
import torch
from scipy.stats import kstest

from mach3sbitools.simulator.priors.cyclical_distribution import CyclicalDistribution


@pytest.fixture(scope="session")
def cyclical_distribution() -> CyclicalDistribution:
    nominals = torch.ones(1)
    lower_bounds = -2 * torch.pi * torch.ones(1, dtype=torch.double)
    upper_bounds = 2 * torch.pi * torch.ones(1, dtype=torch.double)
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
    Compare samples from CyclicalDistribution against a simple
    accept-reject Monte Carlo sampler using the true PDF.
    """
    n_samples = 50_000
    lower, upper = -2 * np.pi, 2 * np.pi
    M = 0.5 / np.pi  # max of the PDF

    accepted = []

    # Generate MC samples via accept-reject
    while len(accepted) < n_samples:
        x = np.random.uniform(lower, upper, size=n_samples)
        y = np.random.uniform(0, M, size=n_samples)

        pdf_vals = (
            cyclical_distribution.pdf(torch.tensor(x, dtype=torch.double).unsqueeze(-1))
            .squeeze()
            .numpy()
        )

        accepted.extend(x[y < pdf_vals])

    accepted = np.array(accepted[:n_samples])

    # Samples from your implementation
    samples = cyclical_distribution.sample(torch.Size([n_samples])).squeeze().numpy()

    # --- Compare distributions ---
    assert abs(samples.mean() - accepted.mean()) < 0.05

    assert abs(samples.var() - accepted.var()) < 0.1

    ks_result = kstest(samples, accepted)
    assert ks_result.pvalue > 0.01
