"""
Integration tests for pyMaCh3Simulator, the Simulator wrapper, and the mach3sbi CLI.

Requires:
  - pyMaCh3_tutorial installed
  - MACH3 env var pointing to a MaCh3 installation containing
    TutorialConfigs/FitterConfig.yaml

All tests are marked with @pytest.mark.mach3_tutorial and @pytest.mark.slow
so they can be excluded from fast CI runs:

    pytest -m "not mach3_tutorial"
"""

import pytest

pytest.importorskip("pyMaCh3_tutorial")

import os
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from mach3sbitools.apps.main_cli import (
    cli,  # adjust import path if your entry-point differs
)
from mach3sbitools.examples.pyMaCh3 import pyMaCh3Simulator
from mach3sbitools.simulator import Simulator, SimulatorProtocol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitter_config() -> Path:
    mach3_root = os.getenv("MACH3", "")
    if not mach3_root:
        pytest.skip("MACH3 environment variable is not set")
    cfg = Path(mach3_root) / "TutorialConfigs" / "FitterConfig.yaml"
    if not cfg.is_file():
        pytest.skip(f"Fitter config not found: {cfg}")
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitter_config() -> Path:
    return _fitter_config()


@pytest.fixture(scope="module")
def pymach3_instance(fitter_config) -> pyMaCh3Simulator:
    """Bare pyMaCh3Simulator (no Simulator wrapper)."""
    return pyMaCh3Simulator(fitter_config)


@pytest.fixture(scope="module")
def simulator_wrapper(fitter_config) -> Simulator:
    """Full Simulator wrapper around pyMaCh3Simulator."""
    return Simulator(
        module_name="mach3sbitools.examples.pyMaCh3",
        class_name="pyMaCh3Simulator",  # note: correct casing
        config=fitter_config,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestProtocolCompliance:
    """Verify pyMaCh3Simulator satisfies SimulatorProtocol at runtime."""

    def test_isinstance_protocol(self, pymach3_instance):
        assert isinstance(pymach3_instance, SimulatorProtocol)

    def test_wrapped_simulator_satisfies_protocol(self, simulator_wrapper):
        assert isinstance(simulator_wrapper.simulator_wrapper, SimulatorProtocol)

    def test_all_protocol_methods_present(self, pymach3_instance):
        required = [
            "simulate",
            "get_parameter_names",
            "get_parameter_bounds",
            "get_is_flat",
            "get_data_bins",
            "get_parameter_nominals",
            "get_parameter_errors",
            "get_covariance_matrix",
            "get_log_likelihood",
        ]
        for method in required:
            assert callable(getattr(pymach3_instance, method, None)), (
                f"Missing method: {method}"
            )


# ---------------------------------------------------------------------------
# Parameter accessors
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestParameterAccessors:
    """Check shapes and basic sanity of all parameter-level accessors."""

    def test_parameter_names_is_list_of_strings(self, pymach3_instance):
        names = pymach3_instance.get_parameter_names()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names), "All names must be strings"

    def test_parameter_bounds_shape(self, pymach3_instance):
        lower, upper = pymach3_instance.get_parameter_bounds()
        n = len(pymach3_instance.get_parameter_names())
        assert len(lower) == n
        assert len(upper) == n

    def test_bounds_ordering(self, pymach3_instance):
        lower, upper = pymach3_instance.get_parameter_bounds()
        assert np.all(np.array(lower) < np.array(upper)), (
            "Every lower bound must be strictly less than its upper bound"
        )

    def test_nominals_within_bounds(self, pymach3_instance):
        lower, upper = pymach3_instance.get_parameter_bounds()
        nominals = pymach3_instance.get_parameter_nominals()
        lower_arr = np.array(lower)
        upper_arr = np.array(upper)
        nom_arr = np.array(nominals)
        assert np.all(nom_arr >= lower_arr) and np.all(nom_arr <= upper_arr), (
            "All nominal values must lie within [lower, upper]"
        )

    def test_errors_positive(self, pymach3_instance):
        errors = pymach3_instance.get_parameter_errors()
        assert np.all(np.array(errors) >= 0), "All errors must be non-negative"

    def test_covariance_matrix_shape(self, pymach3_instance):
        cov = pymach3_instance.get_covariance_matrix()
        n = len(pymach3_instance.get_parameter_names())
        assert cov.shape == (n, n), f"Expected ({n}, {n}), got {cov.shape}"

    def test_covariance_matrix_symmetric(self, pymach3_instance):
        cov = pymach3_instance.get_covariance_matrix()
        np.testing.assert_allclose(
            cov, cov.T, atol=1e-10, err_msg="Covariance matrix must be symmetric"
        )

    def test_covariance_diagonal_non_negative(self, pymach3_instance):
        cov = pymach3_instance.get_covariance_matrix()
        assert np.all(np.diag(cov) >= 0), "Diagonal of covariance must be non-negative"

    def test_data_bins_non_empty(self, pymach3_instance):
        data = pymach3_instance.get_data_bins()
        assert len(data) > 0

    def test_data_bins_non_negative(self, pymach3_instance):
        data = np.array(pymach3_instance.get_data_bins())
        assert np.all(data >= 0), "Data bin counts must be non-negative"

    def test_accessor_lengths_consistent(self, pymach3_instance):
        """All per-parameter accessors must return the same length."""
        n_names = len(pymach3_instance.get_parameter_names())
        lower, upper = pymach3_instance.get_parameter_bounds()
        assert len(pymach3_instance.get_parameter_nominals()) == n_names
        assert len(pymach3_instance.get_parameter_errors()) == n_names
        assert len(lower) == n_names
        assert len(upper) == n_names


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestForwardSimulation:
    """Tests for pyMaCh3Simulator.simulate()."""

    def test_simulate_at_nominal_returns_array(self, pymach3_instance):
        theta = pymach3_instance.get_parameter_nominals()
        x = pymach3_instance.simulate(theta)
        assert hasattr(x, "__len__"), "simulate() must return an array-like"
        assert len(x) == len(pymach3_instance.get_data_bins())

    def test_simulate_output_non_negative(self, pymach3_instance):
        theta = np.array(pymach3_instance.get_parameter_nominals())
        x = pymach3_instance.simulate(theta)
        assert np.all(np.array(x) >= 0), "Simulated bin counts must be non-negative"

    def test_simulate_accepts_list(self, pymach3_instance):
        theta = list(pymach3_instance.get_parameter_nominals())
        x = pymach3_instance.simulate(theta)
        assert len(x) > 0

    def test_simulate_accepts_ndarray(self, pymach3_instance):
        theta = np.array(pymach3_instance.get_parameter_nominals())
        x = pymach3_instance.simulate(theta)
        assert len(x) > 0

    def test_simulate_wrong_length_raises(self, pymach3_instance):
        n = len(pymach3_instance.get_parameter_nominals())
        bad_theta = np.zeros(n + 5)
        with pytest.raises((ValueError, Exception)):
            pymach3_instance.simulate(bad_theta)

    def test_simulate_is_deterministic_at_fixed_theta(self, pymach3_instance):
        """Two calls with the same theta should return the same MC histogram."""
        theta = np.array(pymach3_instance.get_parameter_nominals())
        x1 = np.array(pymach3_instance.simulate(theta))
        x2 = np.array(pymach3_instance.simulate(theta))
        np.testing.assert_array_equal(x1, x2)

    def test_simulate_changes_with_theta(self, pymach3_instance):
        """Perturbing theta should (in general) change the output histogram."""
        theta_nom = np.array(pymach3_instance.get_parameter_nominals())
        errors = np.array(pymach3_instance.get_parameter_errors())
        theta_shifted = theta_nom.copy()
        # Shift first free parameter by 1σ (skip if error is 0)
        nonzero = np.where(errors > 0)[0]
        if len(nonzero) == 0:
            pytest.skip("All parameter errors are zero; cannot test sensitivity")
        theta_shifted[nonzero[0]] += errors[nonzero[0]]
        x_nom = np.array(pymach3_instance.simulate(theta_nom))
        x_shifted = np.array(pymach3_instance.simulate(theta_shifted))
        assert not np.array_equal(x_nom, x_shifted), (
            "Shifting a free parameter should change the simulation output"
        )


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestLogLikelihood:
    """Tests for pyMaCh3Simulator.get_log_likelihood()."""

    def test_returns_float(self, pymach3_instance):
        theta = np.array(pymach3_instance.get_parameter_nominals())
        llh = pymach3_instance.get_log_likelihood(theta)
        assert isinstance(llh, float)

    def test_finite_at_nominal(self, pymach3_instance):
        theta = np.array(pymach3_instance.get_parameter_nominals())
        llh = pymach3_instance.get_log_likelihood(theta)
        assert np.isfinite(llh), "Log-likelihood should be finite at nominal values"


# ---------------------------------------------------------------------------
# Simulator wrapper
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestSimulatorWrapper:
    """Tests for the high-level Simulator class."""

    def test_prior_created(self, simulator_wrapper):
        assert simulator_wrapper.prior is not None

    def test_simulate_returns_two_arrays(self, simulator_wrapper):
        theta, x = simulator_wrapper.simulate(n_samples=3)
        assert theta.ndim == 2, "theta should be 2-D (n_samples, n_params)"
        assert x.ndim == 2, "x should be 2-D (n_samples, n_bins)"

    def test_simulate_sample_count(self, simulator_wrapper):
        n = 5
        theta, x = simulator_wrapper.simulate(n_samples=n)
        # Fewer samples allowed only if some simulations failed
        assert len(theta) <= n
        assert len(x) == len(theta)

    def test_simulate_x_non_negative(self, simulator_wrapper):
        _, x = simulator_wrapper.simulate(n_samples=3)
        assert np.all(x >= 0), "Poisson-smeared counts must be non-negative"

    def test_simulate_theta_within_prior_support(self, simulator_wrapper):
        theta, _ = simulator_wrapper.simulate(n_samples=10)
        lower = simulator_wrapper.prior.prior_data.lower_bounds
        upper = simulator_wrapper.prior.prior_data.upper_bounds
        assert np.all(theta >= lower.cpu().numpy()), (
            "All theta samples must be above lower bounds"
        )
        assert np.all(theta <= upper.cpu().numpy()), (
            "All theta samples must be below upper bounds"
        )

    def test_save_feather(self, simulator_wrapper, tmp_path):
        theta, x = simulator_wrapper.simulate(n_samples=3)
        out = tmp_path / "sims.feather"
        simulator_wrapper.save(out, theta, x)
        assert out.is_file(), "save() must produce a .feather file"
        assert out.stat().st_size > 0

    def test_save_data_parquet(self, simulator_wrapper, tmp_path):
        out = tmp_path / "data.parquet"
        simulator_wrapper.save_data(out)
        assert out.is_file(), "save_data() must produce a parquet file"
        import pyarrow.parquet as pq

        table = pq.read_table(out)
        assert "data" in table.schema.names
        assert len(table["data"]) > 0

    def test_save_prior_pickle(self, simulator_wrapper, tmp_path):
        theta, x = simulator_wrapper.simulate(n_samples=2)
        prior_path = tmp_path / "prior.pkl"
        simulator_wrapper.save(
            tmp_path / "sims.feather", theta, x, prior_path=prior_path
        )
        assert prior_path.is_file(), "Prior pickle file should be created"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mach3_tutorial
class TestCLI:
    """Smoke-tests for mach3sbi CLI subcommands via Click's test runner."""

    _MODULE = "mach3sbitools.examples.pyMaCh3"
    _CLASS = "pyMaCh3Simulator"

    def _base_args(self, fitter_config: Path) -> list[str]:
        return [
            "--simulator_module",
            self._MODULE,
            "--simulator_class",
            self._CLASS,
            "--config",
            str(fitter_config),
        ]

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "mach3sbi" in result.output.lower()

    def test_simulate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0

    def test_save_data_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["save_data", "--help"])
        assert result.exit_code == 0

    def test_create_prior_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["create_prior", "--help"])
        assert result.exit_code == 0

    def test_cli_create_prior(self, fitter_config, tmp_path):
        out = tmp_path / "prior.pkl"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create_prior",
                *self._base_args(fitter_config),
                "--output_file",
                Path(out),
            ],
        )
        assert result.exit_code == 0, f"CLI error:\n{result.output}"
        assert out.is_file(), "create_prior must write a .pkl file"

    def test_cli_save_data(self, fitter_config, tmp_path):
        out = tmp_path / "observed.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save_data",
                *self._base_args(fitter_config),
                "--output_file",
                Path(out),
            ],
        )
        assert result.exit_code == 0, f"CLI error:\n{result.output}"
        assert out.is_file(), "save_data must write a parquet file"

        import pyarrow.parquet as pq

        tbl = pq.read_table(out)
        assert "data" in tbl.schema.names

    def test_cli_simulate(self, fitter_config, tmp_path):
        out = tmp_path / "sims.feather"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "simulate",
                *self._base_args(fitter_config),
                "--output_file",
                str(out),
                "--n_simulations",
                "5",
            ],
        )
        assert result.exit_code == 0, f"CLI error:\n{result.output}"
        assert out.is_file(), "simulate must write a feather file"

    def test_cli_simulate_with_prior_file(self, fitter_config, tmp_path):
        out = tmp_path / "sims.feather"
        prior_out = tmp_path / "prior.pkl"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "simulate",
                *self._base_args(fitter_config),
                "--output_file",
                str(out),
                "--n_simulations",
                "5",
                "--prior_file",
                str(prior_out),
            ],
        )
        assert result.exit_code == 0, f"CLI error:\n{result.output}"
        assert prior_out.is_file(), (
            "Prior file should be written when --prior_file given"
        )

    def test_cli_missing_config_exits_nonzero(self, tmp_path):
        out = tmp_path / "sims.feather"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "simulate",
                "--simulator_module",
                self._MODULE,
                "--simulator_class",
                self._CLASS,
                "--config",
                str(tmp_path / "does_not_exist.yaml"),
                "--output_file",
                str(out),
                "--n_simulations",
                "5",
            ],
        )
        assert result.exit_code != 0 or "error" in result.output.lower(), (
            "Should fail gracefully when config file does not exist"
        )

    def test_cli_bad_class_name_exits_nonzero(self, fitter_config, tmp_path):
        out = tmp_path / "sims.feather"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "simulate",
                "--simulator_module",
                self._MODULE,
                "--simulator_class",
                "NotARealClass",
                "--config",
                str(fitter_config),
                "--output_file",
                str(out),
                "--n_simulations",
                "5",
            ],
        )
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_cli_simulate_with_nuisance_pars(self, fitter_config, tmp_path):
        """Nuisance parameters should be accepted without error."""
        out = tmp_path / "sims.feather"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "simulate",
                *self._base_args(fitter_config),
                "--output_file",
                str(out),
                "--n_simulations",
                "3",
                "--nuisance_pars",
                "syst_*",
            ],
        )
        # We expect either success or a meaningful error — not a traceback crash
        assert result.exit_code in (0, 1), (
            f"Unexpected exit code {result.exit_code}:\n{result.output}"
        )
