"""
Coverage-boosting tests for:
  - simulator/priors/prior.py  (lines 124, 172-176, 207, 278, 296-302, 311-314, 457, 460, 466)
  - simulator/simulator.py     (lines 99-100, 126-127, 140-146)
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import (
    MaskDistributionMap,
    Prior,
    PriorNotFound,
    load_prior,
)
from mach3sbitools.simulator.simulator import Simulator

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
    """All-Gaussian prior with no flat or cyclical params."""
    return Prior(prior_data=_make_prior_data(n))


# ─────────────────────────────────────────────────────────────────────────────
# MaskDistributionMap.to()
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskDistributionMap:
    def test_to_moves_mask_to_device(self):
        """Covers MaskDistributionMap.to() (line 124)."""
        mask = torch.tensor([True, False, True])
        dist = MagicMock()
        mdm = MaskDistributionMap(mask=mask, distribution=dist)
        moved = mdm.to(torch.device("cpu"))
        assert moved.mask.device.type == "cpu"
        assert moved.distribution is dist


# ─────────────────────────────────────────────────────────────────────────────
# Prior properties
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorProperties:
    def test_mean_returns_nominals(self):
        """Covers Prior.mean (line 207)."""
        prior = _gaussian_prior()
        mean = prior.mean
        assert isinstance(mean, torch.Tensor)
        assert mean.shape == (5,)

    def test_n_params(self):
        prior = _gaussian_prior()
        assert prior.n_params == 5

    def test_variance_assembled_from_sub_distributions(self):
        """Covers Prior.variance (lines 296-302)."""
        prior = _gaussian_prior()
        var = prior.variance
        assert var.shape == (5,)
        assert torch.all(var > 0)

    def test_support_returns_constraints(self):
        """Covers Prior.support (lines 311-314)."""
        prior = _gaussian_prior()
        support = prior.support
        assert support is not None

    def test_prior_data_property(self):
        """Covers Prior.prior_data property (line 278)."""
        prior = _gaussian_prior()
        pd_ = prior.prior_data
        assert isinstance(pd_, PriorData)
        assert len(pd_.parameter_names) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Prior sampling — rsample / check_bounds
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorSampling:
    def test_rsample_shape(self):
        """Covers Prior.rsample (lines 172-176) — flat prior supports rsample."""
        data = _make_prior_data(3)
        # Make all params flat so we use Uniform (which supports rsample)
        prior = Prior(prior_data=data, flat_msk=[True, True, True])
        samples = prior.rsample(torch.Size([10]))
        assert samples.shape == (10, 3)

    def test_sample_shape_with_batch(self):
        prior = _gaussian_prior()
        samples = prior.sample(torch.Size([8]))
        assert samples.shape == (8, 5)

    def test_check_bounds_all_in(self):
        prior = _gaussian_prior()
        params = torch.zeros(10, 5)  # all zero — within [-5, 5]
        result = prior.check_bounds(params)
        assert torch.all(result)

    def test_check_bounds_some_out(self):
        prior = _gaussian_prior()
        params = torch.zeros(10, 5)
        params[3, 0] = 100.0  # out of bounds
        result = prior.check_bounds(params)
        assert not result[3]
        assert result.sum() == 9

    def test_gaussian_rejection_sampling_converges(self):
        """Covers rejection-sampling loop inside Prior.sample (line 278 area)."""
        prior = _gaussian_prior(n=3)
        samples = prior.sample(torch.Size([50]))
        assert samples.shape == (50, 3)
        assert torch.all(samples >= prior.prior_data.lower_bounds)
        assert torch.all(samples <= prior.prior_data.upper_bounds)


# ─────────────────────────────────────────────────────────────────────────────
# Prior with nuisance filter
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorNuisanceFilter:
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
# Prior.save / load_prior
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorPersistence:
    def test_save_creates_file(self, tmp_path):
        """Covers Prior.save (line 457)."""
        prior = _gaussian_prior()
        out = tmp_path / "prior.pkl"
        prior.save(out)
        assert out.exists()

    def test_load_prior_round_trip(self, tmp_path):
        """Covers load_prior happy path (line 460, 466)."""
        prior = _gaussian_prior()
        out = tmp_path / "prior.pkl"
        prior.save(out)
        loaded = load_prior(out)
        assert isinstance(loaded, Prior)
        assert loaded.n_params == prior.n_params

    def test_load_prior_raises_not_found(self, tmp_path):
        """Covers PriorNotFound when file doesn't exist (line 460)."""
        with pytest.raises(PriorNotFound):
            load_prior(tmp_path / "nonexistent.pkl")

    def test_load_prior_raises_wrong_type(self, tmp_path):
        """Covers PriorNotFound when file contains wrong type (line 466)."""
        bad = tmp_path / "bad.pkl"
        with bad.open("wb") as f:
            pickle.dump({"not": "a prior"}, f)
        with pytest.raises(PriorNotFound):
            load_prior(bad)

    def test_prior_to_device(self, tmp_path):
        """Covers Prior.to() method."""
        prior = _gaussian_prior()
        result = prior.to(torch.device("cpu"))
        assert result is prior


# ─────────────────────────────────────────────────────────────────────────────
# Prior.to() — move tensors to device
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorCyclical:
    def test_prior_with_cyclical_param(self):
        """Creates a prior with one cyclical parameter — exercises _get_cyclical_map."""
        data = PriorData(
            parameter_names=np.array(["angle", "shift"]),
            nominals=torch.zeros(2),
            covariance_matrix=torch.eye(2),
            lower_bounds=torch.tensor([-2 * torch.pi, -5.0]),
            upper_bounds=torch.tensor([2 * torch.pi, 5.0]),
        )
        prior = Prior(prior_data=data, cyclical_parameters=["angle"])
        samples = prior.sample(torch.Size([10]))
        assert samples.shape == (10, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Simulator error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestSimulatorErrors:
    def test_simulate_skips_bad_simulations(self, dummy_config):
        """Covers exception-catch path in simulate (lines 99-100)."""
        sim = Simulator(
            module_name="dummy_simulator",
            class_name="DummySimulator",
            config=dummy_config,
        )
        # Patch the wrapper to raise on alternate calls
        call_count = [0]
        original = sim.simulator_wrapper.simulate

        def flaky(_):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("bad sim")
            return original(_)

        sim.simulator_wrapper.simulate = flaky
        theta, x = sim.simulate(10)
        # Should have skipped ~5 bad ones
        assert len(theta) < 10
        assert len(theta) == len(x)

    def test_simulate_returns_empty_when_all_bad(self, dummy_config):
        """Covers the empty np.array() fallback (line 100) when valid_x stays None."""
        sim = Simulator(
            module_name="dummy_simulator",
            class_name="DummySimulator",
            config=dummy_config,
        )
        sim.simulator_wrapper.simulate = MagicMock(side_effect=RuntimeError("bad"))
        theta, _ = sim.simulate(5)
        assert len(theta) == 0

    def test_save_with_prior_path(self, dummy_config, tmp_path):
        """Covers the prior_path branch in Simulator.save (lines 126-127)."""
        sim = Simulator(
            module_name="dummy_simulator",
            class_name="DummySimulator",
            config=dummy_config,
        )
        theta = np.ones((5, sim.prior.n_params), dtype=np.float32)
        x = np.ones((5, 12), dtype=np.float32)
        data_file = tmp_path / "data.feather"
        prior_file = tmp_path / "prior.pkl"
        sim.save(data_file, theta, x, prior_path=prior_file)
        assert prior_file.exists()

    def test_save_data_creates_parquet(self, dummy_config, tmp_path):
        """Covers save_data (lines 140-146)."""
        sim = Simulator(
            module_name="dummy_simulator",
            class_name="DummySimulator",
            config=dummy_config,
        )
        out = tmp_path / "obs.parquet"
        sim.save_data(out)
        assert out.exists()

    def test_save_data_accepts_string_path(self, dummy_config, tmp_path):
        """Covers the str->Path conversion in save_data."""
        sim = Simulator(
            module_name="dummy_simulator",
            class_name="DummySimulator",
            config=dummy_config,
        )
        out = str(tmp_path / "obs2.parquet")
        sim.save_data(out)
        assert Path(out).exists()
