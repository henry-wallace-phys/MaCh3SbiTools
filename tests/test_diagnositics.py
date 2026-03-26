"""
Coverage tests for diagnostics/compare_log.py (currently 0%).

Mocks the heavy dependencies (InferenceHandler, Simulator, matplotlib)
so tests run quickly without GPU or real data.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from mach3sbitools.diagnostics import compare_logl
from mach3sbitools.diagnostics.compare_log import normalise_logl

# ── normalise_logl ────────────────────────────────────────────────────────────


class TestNormaliseLogl:
    def test_mean_is_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalise_logl(arr)
        assert abs(result.mean()) < 1e-10

    def test_std_is_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalise_logl(arr)
        assert abs(result.std() - 1.0) < 1e-10

    def test_constant_array_raises_or_nan(self):
        # std=0 → division by zero → np returns nan/inf (not a hard error)
        arr = np.array([3.0, 3.0, 3.0])
        result = normalise_logl(arr)
        assert np.all(np.isnan(result)) or np.all(result == 0)

    def test_output_shape_matches_input(self):
        arr = np.random.rand(50)
        assert normalise_logl(arr).shape == arr.shape


# ── compare_logl (integration smoke test with mocks) ─────────────────────────


def _build_mocks(n_samples: int = 20, n_bins: int = 12):
    """Build minimal mocks for Simulator and InferenceHandler."""
    fake_samples = np.random.randn(n_samples, 5).astype(np.float32)
    fake_llh = np.random.randn(n_samples).astype(np.float64)

    # InferenceHandler mock
    ih = MagicMock()
    sample_mock = MagicMock()
    sample_mock.cpu.return_value.numpy.return_value = fake_samples
    ih.sample_posterior.return_value = sample_mock

    llh_mock = MagicMock()
    llh_mock.cpu.return_value.numpy.return_value = fake_llh
    ih.get_log_likelihood.return_value = llh_mock

    # Simulator mock
    sim = MagicMock()
    sim.simulator_wrapper.get_data_bins.return_value = np.ones(n_bins).tolist()
    sim.simulator_wrapper.get_log_likelihood.return_value = 1.0

    return sim, ih, fake_samples, fake_llh


def _run_compare_logl(sim, ih, **kwargs):
    """Helper to run compare_logl with all heavy/numerical parts mocked."""
    with (
        patch("mach3sbitools.diagnostics.compare_log.plt") as mock_plt,
        patch(
            "mach3sbitools.diagnostics.compare_log.np.polyfit", return_value=(1.0, 0.0)
        ),
    ):
        mock_fig = MagicMock()
        mock_ax2d = MagicMock()
        mock_ax1d = MagicMock()

        mock_plt.subplots.return_value = (mock_fig, (mock_ax2d, mock_ax1d))
        mock_plt.isinteractive.return_value = False
        mock_ax2d.hist2d.return_value = (None, None, None, MagicMock())

        compare_logl(sim, ih, **kwargs)

    return mock_fig, mock_plt


class TestCompareLogl:
    def test_runs_without_save(self):
        """Smoke test: compare_logl runs with no save_path."""
        sim, ih, _, _ = _build_mocks()
        _run_compare_logl(sim, ih, n_samples=20)

        ih.sample_posterior.assert_called_once()
        ih.get_log_likelihood.assert_called_once()

    def test_runs_with_save_path(self, tmp_path):
        """Covers the save_path branch."""
        sim, ih, _, _ = _build_mocks()
        save_path = tmp_path / "compare.png"

        mock_fig, _ = _run_compare_logl(sim, ih, n_samples=20, save_path=save_path)

        mock_fig.savefig.assert_called_once()

    def test_runs_with_likelihood_range(self):
        """Covers the likelihood_range parameter path."""
        sim, ih, _, _ = _build_mocks()

        _run_compare_logl(sim, ih, n_samples=20, likelihood_range=(-3.0, 3.0))

        ih.sample_posterior.assert_called_once()

    def test_runs_with_interactive_mode(self):
        """Covers the plt.isinteractive() == True branch."""
        sim, ih, _, _ = _build_mocks()

        with (
            patch("mach3sbitools.diagnostics.compare_log.plt") as mock_plt,
            patch(
                "mach3sbitools.diagnostics.compare_log.np.polyfit",
                return_value=(1.0, 0.0),
            ),
        ):
            mock_fig = MagicMock()
            mock_ax2d = MagicMock()
            mock_ax1d = MagicMock()

            mock_plt.subplots.return_value = (mock_fig, (mock_ax2d, mock_ax1d))
            mock_plt.isinteractive.return_value = True
            mock_ax2d.hist2d.return_value = (None, None, None, MagicMock())

            compare_logl(sim, ih, n_samples=20)

        mock_fig.show.assert_called_once()

    def test_calls_simulator_log_likelihood_for_each_sample(self):
        """get_log_likelihood should be called once per posterior sample."""
        n_samples = 15
        sim, ih, _, _ = _build_mocks(n_samples=n_samples)

        _run_compare_logl(sim, ih, n_samples=n_samples)

        assert sim.simulator_wrapper.get_log_likelihood.call_count == n_samples


# ── diagnostics __init__ import ───────────────────────────────────────────────


def test_compare_logl_importable():
    """Covers diagnostics/__init__.py lines 1-3."""
    from mach3sbitools.diagnostics import compare_logl as cl

    assert callable(cl)
