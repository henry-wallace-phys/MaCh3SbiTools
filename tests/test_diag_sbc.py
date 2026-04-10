"""
Tests for mach3sbitools.diagnostics.sbc.SBCDiagnostic.

Mocks the heavy sbi/torch dependencies so the suite runs quickly without
GPU or real trained models.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mach3sbitools.diagnostics.sbc import SBCDiagnostic

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

N_PARAMS = 4
N_BINS = 6
N_PRIOR = 5
N_POST = 10


def _make_simulator(n_params: int = N_PARAMS, n_bins: int = N_BINS) -> MagicMock:
    """Minimal Simulator mock."""
    sim = MagicMock()
    sim.simulator_wrapper.simulate.return_value = np.ones(n_bins, dtype=np.float32)
    return sim


def _make_inference_handler(n_params: int = N_PARAMS) -> MagicMock:
    """Minimal InferenceHandler mock whose prior returns sensible tensors."""
    ih = MagicMock()
    # prior.sample must return a float32 tensor of shape (n, n_params)
    ih.prior.sample.return_value = torch.zeros(N_PRIOR, n_params, dtype=torch.float32)
    ih.posterior = MagicMock()
    return ih


def _make_diag(tmp_path: Path) -> SBCDiagnostic:
    """Fully constructed SBCDiagnostic with all heavy deps mocked out."""
    sim = _make_simulator()
    ih = _make_inference_handler()
    return SBCDiagnostic(sim, ih, tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────


class TestSBCDiagnosticInit:
    def test_plot_dir_created(self, tmp_path):
        plot_dir = tmp_path / "plots"
        SBCDiagnostic(_make_simulator(), _make_inference_handler(), plot_dir)
        assert plot_dir.exists()

    def test_prior_samples_none_before_create(self, tmp_path):
        diag = _make_diag(tmp_path)
        assert diag.prior_samples is None
        assert diag.prior_predictives is None

    def test_build_posterior_called_on_init(self, tmp_path):
        ih = _make_inference_handler()
        SBCDiagnostic(_make_simulator(), ih, tmp_path)
        ih.build_posterior.assert_called_once()

    def test_posterior_set_from_handler(self, tmp_path):
        ih = _make_inference_handler()
        diag = SBCDiagnostic(_make_simulator(), ih, tmp_path)
        assert diag.posterior is ih.posterior


# ─────────────────────────────────────────────────────────────────────────────
# create_prior_samples
# ─────────────────────────────────────────────────────────────────────────────


class TestCreatePriorSamples:
    def test_prior_samples_shape(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert diag.prior_samples.shape == (N_PRIOR, N_PARAMS)

    def test_prior_predictives_shape(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert diag.prior_predictives.shape == (N_PRIOR, N_BINS)

    def test_simulator_called_once_per_sample(self, tmp_path):
        sim = _make_simulator()
        ih = _make_inference_handler()
        diag = SBCDiagnostic(sim, ih, tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert sim.simulator_wrapper.simulate.call_count == N_PRIOR

    def test_prior_sampled_with_correct_count(self, tmp_path):
        ih = _make_inference_handler()
        diag = SBCDiagnostic(_make_simulator(), ih, tmp_path)
        diag.create_prior_samples(N_PRIOR)
        ih.prior.sample.assert_called_once_with((N_PRIOR,))

    def test_prior_predictives_dtype_float32(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert diag.prior_predictives.dtype == torch.float32

    def test_prior_samples_dtype_float32(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert diag.prior_samples.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# _check_prior_sampled guard
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckPriorSampled:
    def test_rank_plot_raises_if_not_sampled(self, tmp_path):
        diag = _make_diag(tmp_path)
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.rank_plot()

    def test_expected_coverage_raises_if_not_sampled(self, tmp_path):
        diag = _make_diag(tmp_path)
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.expected_coverage()

    def test_tarp_raises_if_not_sampled(self, tmp_path):
        diag = _make_diag(tmp_path)
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.tarp()


# ─────────────────────────────────────────────────────────────────────────────
# rank_plot
# ─────────────────────────────────────────────────────────────────────────────


class TestRankPlot:
    def _patched_run(self, tmp_path, **rank_plot_kwargs):
        """Run rank_plot with sbi internals fully mocked."""
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)

        fake_ranks = torch.zeros(N_PRIOR, N_PARAMS)
        fake_fig = MagicMock()

        with (
            patch(
                "mach3sbitools.diagnostics.sbc.run_sbc",
                return_value=(fake_ranks, MagicMock()),
            ) as mock_run_sbc,
            patch(
                "mach3sbitools.diagnostics.sbc.sbc_rank_plot",
                return_value=(fake_fig, MagicMock()),
            ) as mock_rank_plot,
            patch("mach3sbitools.diagnostics.sbc.plt") as mock_plt,
        ):
            diag.rank_plot(num_posterior_samples=N_POST, **rank_plot_kwargs)

        return mock_run_sbc, mock_rank_plot, mock_plt, fake_fig

    def test_run_sbc_called(self, tmp_path):
        mock_run_sbc, *_ = self._patched_run(tmp_path)
        mock_run_sbc.assert_called_once()

    def test_sbc_rank_plot_called(self, tmp_path):
        _, mock_rank_plot, *_ = self._patched_run(tmp_path)
        mock_rank_plot.assert_called_once()

    def test_figure_saved(self, tmp_path):
        *_, fake_fig = self._patched_run(tmp_path)
        fake_fig.savefig.assert_called_once()
        saved_path = fake_fig.savefig.call_args[0][0]
        assert str(saved_path).endswith("rank_plot.pdf")

    def test_figure_closed(self, tmp_path):
        *_, mock_plt, _ = self._patched_run(tmp_path)
        mock_plt.close.assert_called_once()

    def test_num_posterior_samples_forwarded(self, tmp_path):
        mock_run_sbc, *_ = self._patched_run(tmp_path)
        _, call_kwargs = mock_run_sbc.call_args
        assert call_kwargs.get("num_posterior_samples") == N_POST

    def test_run_sbc_receives_prior_tensors(self, tmp_path):
        mock_run_sbc, *_ = self._patched_run(tmp_path)
        args, _ = mock_run_sbc.call_args
        # positional args: prior_samples, prior_predictives, posterior
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)

    def test_custom_num_rank_bins_forwarded(self, tmp_path):
        _, mock_rank_plot, *_ = self._patched_run(tmp_path, num_rank_bins=30)
        _, call_kwargs = mock_rank_plot.call_args
        assert call_kwargs.get("num_bins") == 30


# ─────────────────────────────────────────────────────────────────────────────
# expected_coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestExpectedCoverage:
    def _patched_run(self, tmp_path, **kwargs):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)

        fake_ranks = torch.zeros(N_PRIOR, N_PARAMS)
        fake_fig = MagicMock()

        with (
            patch(
                "mach3sbitools.diagnostics.sbc.run_sbc",
                return_value=(fake_ranks, MagicMock()),
            ) as mock_run_sbc,
            patch(
                "mach3sbitools.diagnostics.sbc.sbc_rank_plot",
                return_value=(fake_fig, MagicMock()),
            ) as mock_rank_plot,
            patch("mach3sbitools.diagnostics.sbc.plt") as mock_plt,
        ):
            diag.expected_coverage(num_posterior_samples=N_POST, **kwargs)

        return mock_run_sbc, mock_rank_plot, mock_plt, fake_fig

    def test_run_sbc_called(self, tmp_path):
        mock_run_sbc, *_ = self._patched_run(tmp_path)
        mock_run_sbc.assert_called_once()

    def test_plot_type_is_cdf(self, tmp_path):
        _, mock_rank_plot, *_ = self._patched_run(tmp_path)
        _, call_kwargs = mock_rank_plot.call_args
        assert call_kwargs.get("plot_type") == "cdf"

    def test_figure_saved(self, tmp_path):
        *_, fake_fig = self._patched_run(tmp_path)
        fake_fig.savefig.assert_called_once()
        saved_path = fake_fig.savefig.call_args[0][0]
        assert str(saved_path).endswith("expected_coverage.pdf")

    def test_figure_closed(self, tmp_path):
        *_, mock_plt, _ = self._patched_run(tmp_path)
        mock_plt.close.assert_called_once()

    def test_reduce_fn_passed_to_run_sbc(self, tmp_path):
        """expected_coverage passes a reduce_fns kwarg; rank_plot does not."""
        mock_run_sbc, *_ = self._patched_run(tmp_path)
        _, call_kwargs = mock_run_sbc.call_args
        assert "reduce_fns" in call_kwargs

    def test_raises_if_posterior_none(self, tmp_path):
        """If posterior is None after sampling, ValueError is raised."""
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        diag.posterior = None
        with pytest.raises(ValueError, match="Posterior predictives not set"):
            diag.expected_coverage()


# ─────────────────────────────────────────────────────────────────────────────
# tarp
# ─────────────────────────────────────────────────────────────────────────────


class TestTarp:
    def _patched_run(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)

        fake_ecp = torch.linspace(0, 1, 10)
        fake_alpha = torch.linspace(0, 1, 10)
        fake_fig = MagicMock()

        with (
            patch(
                "mach3sbitools.diagnostics.sbc.run_tarp",
                return_value=(fake_ecp, fake_alpha),
            ) as mock_run_tarp,
            patch(
                "mach3sbitools.diagnostics.sbc.check_tarp",
                return_value=(0.01, 0.95),
            ) as mock_check_tarp,
            patch(
                "mach3sbitools.diagnostics.sbc.plot_tarp",
                return_value=(fake_fig, MagicMock()),
            ) as mock_plot_tarp,
            patch("mach3sbitools.diagnostics.sbc.plt") as mock_plt,
        ):
            diag.tarp(num_posterior_samples=N_POST)

        return mock_run_tarp, mock_check_tarp, mock_plot_tarp, mock_plt, fake_fig

    def test_run_tarp_called(self, tmp_path):
        mock_run_tarp, *_ = self._patched_run(tmp_path)
        mock_run_tarp.assert_called_once()

    def test_check_tarp_called(self, tmp_path):
        _, mock_check_tarp, *_ = self._patched_run(tmp_path)
        mock_check_tarp.assert_called_once()

    def test_plot_tarp_called(self, tmp_path):
        _, _, mock_plot_tarp, *_ = self._patched_run(tmp_path)
        mock_plot_tarp.assert_called_once()

    def test_figure_saved(self, tmp_path):
        *_, fake_fig = self._patched_run(tmp_path)
        fake_fig.savefig.assert_called_once()
        saved_path = fake_fig.savefig.call_args[0][0]
        assert str(saved_path).endswith("tarp.pdf")

    def test_figure_closed(self, tmp_path):
        *_, mock_plt, _ = self._patched_run(tmp_path)
        mock_plt.close.assert_called_once()

    def test_num_posterior_samples_forwarded(self, tmp_path):
        mock_run_tarp, *_ = self._patched_run(tmp_path)
        _, call_kwargs = mock_run_tarp.call_args
        assert call_kwargs.get("num_posterior_samples") == N_POST

    def test_references_none_by_default(self, tmp_path):
        mock_run_tarp, *_ = self._patched_run(tmp_path)
        _, call_kwargs = mock_run_tarp.call_args
        assert call_kwargs.get("references") is None
