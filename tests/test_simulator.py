import numpy as np
import pytest
from scipy import stats

from mach3sbitools.simulator.simulator import Simulator
from mach3sbitools.utils import from_feather


@pytest.fixture
def simulator(dummy_config):
    return Simulator(
        module_name="dummy_simulator",
        class_name="DummySimulator",
        config=dummy_config,
        cyclical_pars=["theta_9"],
    )


@pytest.mark.slow
def test_simulate_returns_at_most_n_samples(simulator):
    """Since bad simulations are skipped, we should get at most n_sims back"""
    n_sims = 5000
    theta, x = simulator.simulate(n_sims)

    assert theta.shape[0] <= n_sims
    assert x.shape[0] <= n_sims
    assert theta.shape[0] == x.shape[0]


@pytest.mark.slow
def test_simulate_x_is_poisson_distributed(simulator):
    """
    DummySimulator.simulate returns np.ones(10), so x_sample ~ Poisson(lambda=1).
    We use a chi-squared goodness-of-fit test on each output bin.
    """
    n_sims = 200000
    _, x = simulator.simulate(n_sims)

    # Test each output bin independently
    for bin_idx in range(x.shape[1]):
        bin_values = x[:, bin_idx]

        # Poisson(1): test counts for k=0,1,2,3,4+
        max_k = 5
        observed_counts = np.array(
            [np.sum(bin_values == k) for k in range(max_k)]
            + [np.sum(bin_values >= max_k)]
        )

        expected_probs = np.array(
            [stats.poisson.pmf(k, mu=1) for k in range(max_k)]
            + [1 - stats.poisson.cdf(max_k - 1, mu=1)]
        )
        expected_counts = expected_probs * n_sims

        # Chi-squared goodness of fit
        # Only include bins with expected count >= 5 to keep the test valid
        valid = expected_counts >= 5
        chi2, p_value = stats.chisquare(
            observed_counts[valid], f_exp=expected_counts[valid]
        )

        assert p_value > 0.001, (
            f"Bin {bin_idx} failed Poisson(1) chi-squared test: "
            f"chi2={chi2:.3f}, p={p_value:.4f}"
        )


@pytest.mark.slow
def test_simulate_x_is_non_negative_integer(simulator):
    """Poisson samples must be non-negative integers"""
    _, x = simulator.simulate(10000)

    assert np.all(x >= 0), "All Poisson samples should be non-negative"
    assert np.all(x == x.astype(int)), "All Poisson samples should be integers"


@pytest.mark.slow
def test_simulate_x_mean_close_to_one(simulator):
    """Poisson(1) has mean=1 and variance=1. Check within 3 sigma."""
    n_sims = 20000
    _, x = simulator.simulate(n_sims)

    # Standard error of the mean for Poisson(1) over n_sims samples
    expected_mean = 1.0
    col_means = x.mean(axis=0)
    assert np.all(np.abs(col_means - expected_mean) < 0.05), (
        f"Some bin means are >3σ from expected 1.0: {col_means}"
    )


@pytest.mark.slow
def test_simulate_x_variance_close_to_one(simulator):
    """Poisson(1) has variance=1. Check within reasonable tolerance."""
    n_sims = 200000
    _, x = simulator.simulate(n_sims)

    col_vars = x.var(axis=0)
    assert np.all(np.abs(col_vars - 1.0) < 0.05), (
        f"Some bin variances are far from expected 1.0: {col_vars}"
    )


@pytest.mark.slow
def test_save(simulator, tmp_path):
    n_sims = 10
    x, t = simulator.simulate(n_sims)

    data_file = tmp_path / "data.feather"
    simulator.save(data_file, t, x)

    assert data_file.exists()

    x, t = from_feather(data_file, simulator.prior.prior_data.parameter_names.tolist())
    assert len(x) == len(t) == 10
