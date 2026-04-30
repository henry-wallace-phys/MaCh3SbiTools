"""
Tests for mach3sbitools.diagnostics.sbc.SBCDiagnostic.

Keeps one integration-style smoke test per diagnostic method (rank_plot,
expected_coverage, tarp) rather than asserting each mock call individually.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mach3sbitools.diagnostics.sbc import SBCDiagnostic

N_PARAMS, N_BINS, N_PRIOR, N_POST = 4, 6, 5, 10


def _make_diag(tmp_path: Path) -> SBCDiagnostic:
    sim = MagicMock()
    sim.simulator_wrapper.simulate.return_value = np.ones(N_BINS, dtype=np.float32)
    ih = MagicMock()
    ih.prior.sample.return_value = torch.zeros(N_PRIOR, N_PARAMS, dtype=torch.float32)
    ih.posterior = MagicMock()
    return SBCDiagnostic(sim, ih, tmp_path)


class TestSBCDiagnostic:
    def test_raises_before_prior_sampled(self, tmp_path):
        diag = _make_diag(tmp_path)
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.rank_plot()
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.expected_coverage()
        with pytest.raises(ValueError, match="Prior predictives not set"):
            diag.tarp()

    def test_create_prior_samples_populates_tensors(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        assert diag.prior_samples.shape == (N_PRIOR, N_PARAMS)
        assert diag.prior_predictives.shape == (N_PRIOR, N_BINS)
        assert diag.prior_samples.dtype == torch.float32

    def test_rank_plot_writes_pdf(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        fake_fig = MagicMock()
        with (
            patch(
                "mach3sbitools.diagnostics.sbc.run_sbc",
                return_value=(torch.zeros(N_PRIOR, N_PARAMS), MagicMock()),
            ),
            patch(
                "mach3sbitools.diagnostics.sbc.sbc_rank_plot",
                return_value=(fake_fig, MagicMock()),
            ),
            patch("mach3sbitools.diagnostics.sbc.plt"),
        ):
            diag.rank_plot(num_posterior_samples=N_POST)
        saved_path = fake_fig.savefig.call_args[0][0]
        assert str(saved_path).endswith("rank_plot.pdf")

    def test_expected_coverage_writes_pdf_with_cdf_plot_type(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        fake_fig = MagicMock()
        with (
            patch(
                "mach3sbitools.diagnostics.sbc.run_sbc",
                return_value=(torch.zeros(N_PRIOR, N_PARAMS), MagicMock()),
            ),
            patch(
                "mach3sbitools.diagnostics.sbc.sbc_rank_plot",
                return_value=(fake_fig, MagicMock()),
            ) as mock_plot,
            patch("mach3sbitools.diagnostics.sbc.plt"),
        ):
            diag.expected_coverage(num_posterior_samples=N_POST)
        assert mock_plot.call_args[1].get("plot_type") == "cdf"
        assert str(fake_fig.savefig.call_args[0][0]).endswith("expected_coverage.pdf")

    def test_tarp_writes_pdf(self, tmp_path):
        diag = _make_diag(tmp_path)
        diag.create_prior_samples(N_PRIOR)
        fake_fig = MagicMock()
        ecp = torch.linspace(0, 1, 10)
        alpha = torch.linspace(0, 1, 10)
        with (
            patch("mach3sbitools.diagnostics.sbc.run_tarp", return_value=(ecp, alpha)),
            patch(
                "mach3sbitools.diagnostics.sbc.check_tarp", return_value=(0.01, 0.95)
            ),
            patch(
                "mach3sbitools.diagnostics.sbc.plot_tarp",
                return_value=(fake_fig, MagicMock()),
            ),
            patch("mach3sbitools.diagnostics.sbc.plt"),
        ):
            diag.tarp(num_posterior_samples=N_POST)
        assert str(fake_fig.savefig.call_args[0][0]).endswith("tarp.pdf")
