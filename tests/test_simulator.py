"""
Tests for mach3sbitools.simulator.

Covers Simulator (forward simulation, persistence), Prior (construction,
sampling, properties, persistence), and related helpers.

Merged from the former test_simulator.py and test_simulator_coverage.py,
which had an artificial split between "behavioural" and "coverage" tests
for the same modules.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import (
    MaskDistributionMap,
    Prior,
    PriorNotFound,
    _check_boundary,
    load_prior,
)
from mach3sbitools.simulator.simulator import Simulator
from mach3sbitools.utils import from_feather, get_logger

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_prior_data(n: int = 5) -> PriorData:
    return PriorData(
        parameter_names=np.array([f"p{i}" for i in range(n)]),
        nominals=torch.ones(n),
        covariance_matrix=torch.eye(n),
        lower_bounds=torch.full((n,), -5.0),
        upper_bounds=torch.full((n,), 5.0),
    )


def _gaussian_prior(n: int = 5) -> Prior:
    return Prior(prior_data=_make_prior_data(n))


@pytest.fixture
def simulator(dummy_config):
    return Simulator(
        module_name="dummy_simulator",
        class_name="DummySimulator",
        config=dummy_config,
        cyclical_pars=["theta_9"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# MaskDistributionMap
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskDistributionMap:
    def test_to_moves_mask_to_device(self):
        mask = torch.tensor([True, False, True])
        dist = MagicMock()
        moved = MaskDistributionMap(mask=mask, distribution=dist).to(
            torch.device("cpu")
        )
        assert moved.mask.device.type == "cpu"
        assert moved.distribution is dist


# ─────────────────────────────────────────────────────────────────────────────
# Prior — construction and properties
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorProperties:
    def test_dist_types_from_conftest_prior(self, prior):
        """prior fixture has cyclical + flat + gaussian components."""
        assert len(prior._priors) == 3
        assert torch.sum(prior._priors[0].mask).item() == 1  # cyclical
        assert torch.sum(prior._priors[1].mask).item() == 3  # flat
        assert torch.sum(prior._priors[2].mask).item() == 26  # gaussian

    def test_mean_returns_nominals(self):
        prior = _gaussian_prior()
        mean = prior.mean
        assert isinstance(mean, torch.Tensor)
        assert mean.shape == (5,)

    def test_n_params(self):
        assert _gaussian_prior().n_params == 5

    def test_variance_shape_and_positive(self):
        var = _gaussian_prior().variance
        assert var.shape == (5,)
        assert torch.all(var > 0)

    def test_support_not_none(self):
        assert _gaussian_prior().support is not None

    def test_prior_data_property(self):
        pd_ = _gaussian_prior().prior_data
        assert isinstance(pd_, PriorData)
        assert len(pd_.parameter_names) == 5

    def test_nuisance_filter_removes_params(self):
        data = PriorData(
            parameter_names=np.array(["keep_a", "drop_x", "keep_b"]),
            nominals=torch.ones(3),
            covariance_matrix=torch.eye(3),
            lower_bounds=torch.full((3,), -5.0),
            upper_bounds=torch.full((3,), 5.0),
        )
        prior = Prior(prior_data=data, nuisance_parameters=["drop_*"])
        assert prior.n_params == 2
        assert "drop_x" not in prior.prior_data.parameter_names


# ─────────────────────────────────────────────────────────────────────────────
# Prior — sampling
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorSampling:
    @pytest.mark.parametrize(
        "shape,expected",
        [
            (torch.Size([8]), (8, 5)),
            (torch.Size([1]), (1, 5)),
        ],
    )
    def test_sample_shape(self, shape, expected):
        assert _gaussian_prior().sample(shape).shape == torch.Size(expected)

    def test_rsample_shape_with_flat_prior(self):
        """Uniform supports rsample; use all-flat prior to exercise that path."""
        data = _make_prior_data(3)
        prior = Prior(prior_data=data, flat_msk=[True, True, True])
        assert prior.rsample(torch.Size([10])).shape == (10, 3)

    def test_sample_within_bounds(self):
        prior = _gaussian_prior()
        samples = prior.sample(torch.Size([50]))
        assert torch.all(samples >= prior.prior_data.lower_bounds)
        assert torch.all(samples <= prior.prior_data.upper_bounds)

    def test_check_bounds_all_inside(self):
        prior = _gaussian_prior()
        assert torch.all(prior.check_bounds(torch.zeros(10, 5)))

    def test_check_bounds_one_outside(self):
        prior = _gaussian_prior()
        params = torch.zeros(10, 5)
        params[3, 0] = 100.0
        result = prior.check_bounds(params)
        assert not result[3]
        assert result.sum() == 9

    def test_cyclical_prior_samples_correct_shape(self):
        data = PriorData(
            parameter_names=np.array(["angle", "shift"]),
            nominals=torch.zeros(2),
            covariance_matrix=torch.eye(2),
            lower_bounds=torch.tensor([-2 * torch.pi, -5.0]),
            upper_bounds=torch.tensor([2 * torch.pi, 5.0]),
        )
        prior = Prior(prior_data=data, cyclical_parameters=["angle"])
        assert prior.sample(torch.Size([10])).shape == (10, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PriorData — slicing
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorDataSlicing:
    def test_slicing_returns_correct_subset(self):
        param_names = np.array(["a", "b", "c"])
        nominals = torch.Tensor([1, 2, 3])
        cov = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        lb = torch.Tensor([1, 2, 3])
        ub = torch.Tensor([4, 5, 6])

        data = PriorData(param_names, nominals, cov, lb, ub)
        mask = torch.tensor([True, False, True])
        sliced = data[mask]

        np.testing.assert_array_equal(
            sliced.parameter_names, param_names[[True, False, True]]
        )
        assert torch.equal(sliced.nominals, nominals[mask])
        assert torch.equal(sliced.covariance_matrix, cov[mask][:, mask])


# ─────────────────────────────────────────────────────────────────────────────
# Prior — persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        prior = _gaussian_prior()
        out = tmp_path / "prior.pkl"
        prior.save(out)
        loaded = load_prior(out)
        assert isinstance(loaded, Prior)
        assert loaded.n_params == prior.n_params

    def test_load_prior_raises_not_found(self, tmp_path):
        with pytest.raises(PriorNotFound):
            load_prior(tmp_path / "nonexistent.pkl")

    def test_load_prior_raises_wrong_type(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        with bad.open("wb") as f:
            pickle.dump({"not": "a prior"}, f)
        with pytest.raises(PriorNotFound):
            load_prior(bad)

    def test_prior_to_device_returns_self(self):
        prior = _gaussian_prior()
        assert prior.to(torch.device("cpu")) is prior


# ─────────────────────────────────────────────────────────────────────────────
# _check_boundary warnings
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckBoundary:
    """Shared inputs for both warning tests."""

    @pytest.fixture()
    def _inputs(self):

        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.1, 0.1]),
            np.array(["param_a", "param_b"]),
            get_logger(),
        )

    def test_warning_emitted_for_wide_bounds(self, _inputs):

        nominal, error, names, logger = _inputs
        lower = torch.tensor([-100.0, 2.0])
        upper = torch.tensor([1.0, 100.0])

        with patch.object(logger, "warning") as mock_warning:
            _check_boundary(nominal, error, lower, upper, names)
            assert mock_warning.call_count == 3  # header + 2 params

    def test_no_warning_for_tight_bounds(self, _inputs):

        nominal, error, names, logger = _inputs
        lower = torch.tensor([0.5, 1.5])
        upper = torch.tensor([1.5, 2.5])

        with patch.object(logger, "warning") as mock_warning:
            _check_boundary(nominal, error, lower, upper, names)
        mock_warning.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Simulator — forward simulation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSimulatorForward:
    # def test_returns_at_most_n_samples(self, simulator):
    #     n = 5000
    #     theta, x = simulator.simulate(n)
    #     assert theta.shape[0] <= n
    #     assert theta.shape[0] == x.shape[0]

    # def test_x_is_poisson_distributed(self, simulator):
    #     """Each output bin should follow Poisson(lambda=1)."""
    #     n = 20_000
    #     _, x = simulator.simulate(n)
    #     max_k = 5
    #     for bin_idx in range(x.shape[1]):
    #         vals = x[:, bin_idx]
    #         obs = np.array(
    #             [np.sum(vals == k) for k in range(max_k)] + [np.sum(vals >= max_k)]
    #         )
    #         probs = np.array(
    #             [stats.poisson.pmf(k, mu=1) for k in range(max_k)]
    #             + [1 - stats.poisson.cdf(max_k - 1, mu=1)]
    #         )
    #         exp = probs * n
    #         valid = exp >= 5
    #         _, p = stats.chisquare(obs[valid], f_exp=exp[valid])
    #         assert p > 0.001, f"Bin {bin_idx}: chi2 p={p:.4f}"

    def test_x_non_negative_integers(self, simulator):
        _, x = simulator.simulate(1000)
        assert np.all(x >= 0)
        assert np.all(x == x.astype(int))

    def test_x_mean_near_one(self, simulator):
        _, x = simulator.simulate(20_000)
        assert np.all(np.abs(x.mean(axis=0) - 1.0) < 0.05)

    def test_x_variance_near_one(self, simulator):
        _, x = simulator.simulate(200_000)
        assert np.all(np.abs(x.var(axis=0) - 1.0) < 0.05)

    def test_skips_bad_simulations(self, dummy_config):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        call_count = [0]
        original = sim.simulator_wrapper.simulate

        def flaky(_):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("bad sim")
            return original(_)

        sim.simulator_wrapper.simulate = flaky
        theta, x = sim.simulate(10)
        assert len(theta) < 10
        assert len(theta) == len(x)

    def test_returns_empty_when_all_bad(self, dummy_config):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        sim.simulator_wrapper.simulate = MagicMock(side_effect=RuntimeError("bad"))
        theta, _ = sim.simulate(5)
        assert len(theta) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Simulator — persistence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSimulatorPersistence:
    def test_save_feather_round_trip(self, simulator, tmp_path):
        theta, x = simulator.simulate(10)
        path = tmp_path / "data.feather"
        simulator.save(path, theta, x)
        assert path.exists()
        t, x2 = from_feather(path, simulator.prior.prior_data.parameter_names.tolist())
        assert len(t) == len(x2) == 10

    def test_save_with_prior_path(self, dummy_config, tmp_path):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        theta = np.ones((5, sim.prior.n_params), dtype=np.float32)
        x = np.ones((5, 12), dtype=np.float32)
        prior_file = tmp_path / "prior.pkl"
        sim.save(tmp_path / "data.feather", theta, x, prior_path=prior_file)
        assert prior_file.exists()

    def test_save_data_creates_parquet(self, dummy_config, tmp_path):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        out = tmp_path / "obs.parquet"
        sim.save_data(out)
        assert out.exists()

    def test_save_data_accepts_string_path(self, dummy_config, tmp_path):
        sim = Simulator("dummy_simulator", "DummySimulator", config=dummy_config)
        out = str(tmp_path / "obs2.parquet")
        sim.save_data(out)
        assert Path(out).exists()
