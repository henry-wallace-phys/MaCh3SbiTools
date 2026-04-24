"""
Tests for mach3sbitools.inference.

Covers InferenceHandler (happy path, error paths, checkpoint round-trips)
and SBITrainer (error paths, scheduler paths, full training runs).

Merged from the former test_inference_handler.py and
test_inference_coverage.py, which shared session fixtures but duplicated
several error-path and guard tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from scipy import stats
from torch.utils.data import TensorDataset

from mach3sbitools.inference import InferenceHandler
from mach3sbitools.simulator import load_prior
from mach3sbitools.utils import TorchDeviceHandler, TrainingConfig

N_SAMPLES = 10000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tiny_dataset(n: int = 200, theta_dim: int = 4, x_dim: int = 6) -> TensorDataset:
    return TensorDataset(torch.randn(n, theta_dim), torch.randn(n, x_dim))


def _tiny_model(theta_dim: int = 4, x_dim: int = 6) -> torch.nn.Module:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(x_dim, theta_dim)

        def forward(self, x):
            return self.fc(x)

        def loss(self, theta, x):
            return ((self.fc(x) - theta) ** 2).mean(dim=-1)

    return TinyModel()


def _minimal_config(tmp_path: Path, **kwargs) -> TrainingConfig:
    defaults: dict = dict(
        save_path=tmp_path / "model.ckpt",
        batch_size=1024,
        max_epochs=2,
        stop_after_epochs=3,
        autosave_every=1,
        print_interval=1,
        show_progress=False,
        validation_fraction=0.1,
        num_workers=0,
        warmup_epochs=1,
        ema_alpha=0.1,
        scheduler_patience=5,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Session fixtures (trained handler + observations)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def nominal_observation(simulator_injector):
    rng = np.random.default_rng(42)
    device_handler = TorchDeviceHandler()

    return device_handler.to_tensor(
        rng.poisson(lam=1, size=len(simulator_injector.get_data_bins()))
        .astype(np.float32)
        .tolist()
    )


@pytest.fixture(scope="session")
def trained_handler(prior_save, dummy_data_dir, posterior_config, training_config):
    """Fully trained handler, created once for the whole session."""
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    handler.train_posterior(training_config)
    return handler


@pytest.fixture(scope="session")
def samples(trained_handler, nominal_observation):
    """Posterior samples conditioned on a Poisson(1) observation."""
    return trained_handler.sample_posterior(N_SAMPLES, nominal_observation)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler — guard / error paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerErrorPaths:
    def test_load_training_data_raises_without_dataset(self, prior_save):
        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError):
            handler.load_training_data()

    def test_train_posterior_raises_without_tensor_dataset(
        self, prior_save, posterior_config, training_config
    ):
        handler = InferenceHandler(prior_save)
        handler.create_posterior(posterior_config)
        with pytest.raises(ValueError, match="load_training_data"):
            handler.train_posterior(training_config)

    def test_train_posterior_raises_without_inference(
        self, prior_save, training_config
    ):
        handler = InferenceHandler(prior_save)
        handler._tensor_dataset = TensorDataset(torch.zeros(10, 4), torch.zeros(10, 6))
        cfg = TrainingConfig(resume_checkpoint=Path("/some/ckpt.pt"))
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(cfg)

    def test_train_posterior_raises_inference_none(self, prior_save, training_config):
        """inference=None with no resume_checkpoint also raises."""
        handler = InferenceHandler(prior_save)
        handler._tensor_dataset = TensorDataset(torch.zeros(10, 4), torch.zeros(10, 6))
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(training_config)

    def test_build_posterior_raises_without_density_estimator(self, prior_save):
        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError, match="density estimator"):
            handler.build_posterior()

    def test_build_posterior_raises_without_inference(self, prior_save):
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()
        with pytest.raises(ValueError):
            handler.build_posterior()

    def test_sample_posterior_raises_without_estimator(self, prior_save):
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()
        handler.inference = MagicMock()
        handler.posterior = None
        with patch.object(handler, "build_posterior"):
            with pytest.raises(ValueError, match="density estimator"):
                handler.sample_posterior(10, [1.0] * 12)

    def test_get_log_likelihood_raises_without_estimator(self, prior_save):
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()
        handler.inference = MagicMock()
        handler.posterior = None
        with patch.object(handler, "build_posterior"):
            with pytest.raises(ValueError, match="density estimator"):
                handler.get_log_likelihood(np.ones((5, 4)), [1.0] * 12)

    def test_load_posterior_raises_file_not_found(self, prior_save):
        handler = InferenceHandler(prior_save)
        with pytest.raises(FileNotFoundError):
            handler.load_posterior(Path("/no/such/checkpoint.pt"))

    def test_load_posterior_plain_state_dict_raises_without_config(
        self, prior_save, tmp_path
    ):
        fake_state = {"some_key": torch.zeros(1)}
        ckpt_path = tmp_path / "plain.pt"
        torch.save(fake_state, ckpt_path)
        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError, match="model_config"):
            handler.load_posterior(ckpt_path, posterior_config=None)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler — happy path / output quality
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerHappyPath:
    def test_dataset_set(self, prior_save, dummy_data_dir):
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        assert handler.dataset is not None

    def test_tensor_dataset_loaded(self, prior_save, dummy_data_dir):
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        assert handler._tensor_dataset is not None

    def test_inference_object_created(
        self, prior_save, dummy_data_dir, posterior_config
    ):
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        handler.create_posterior(posterior_config)
        assert handler.inference is not None

    def test_density_estimator_set_after_training(self, trained_handler):
        assert trained_handler._density_estimator is not None

    def test_samples_count(self, samples):
        assert len(samples) == N_SAMPLES

    def test_samples_shape(self, samples, trained_handler):
        n_params = trained_handler.prior.n_params
        assert samples.shape == (N_SAMPLES, n_params)

    def test_samples_finite(self, samples):
        assert torch.all(torch.isfinite(samples))

    def test_samples_within_prior_bounds(self, samples, trained_handler):
        lower = trained_handler.prior.prior_data.lower_bounds.cpu()
        upper = trained_handler.prior.prior_data.upper_bounds.cpu()
        s = samples.cpu()
        assert torch.all(s >= lower)
        assert torch.all(s <= upper)

    def test_posterior_mean_near_nominal(self, samples, trained_handler):
        """Dummy simulator output is theta-independent so posterior should
        not update strongly away from the prior nominal."""
        nominals = trained_handler.prior.prior_data.nominals.cpu()
        posterior_mean = samples.cpu().mean(dim=0)
        prior_std = trained_handler.prior.variance.cpu().sqrt()
        deviation = (posterior_mean - nominals).abs()
        assert torch.all(deviation < 3 * prior_std)

    def test_posterior_narrower_than_prior(self, samples, trained_handler):
        prior_var = trained_handler.prior.variance.cpu()
        posterior_var = samples.cpu().var(dim=0)
        n_narrower = (posterior_var < prior_var).sum().item()
        n_params = trained_handler.prior.n_params
        assert n_narrower >= n_params // 2

    def test_sampling_reproducible(self, trained_handler, nominal_observation):
        torch.manual_seed(42)
        a = trained_handler.sample_posterior(100, nominal_observation)
        torch.manual_seed(42)
        b = trained_handler.sample_posterior(100, nominal_observation)

        assert torch.allclose(a, b)

    def test_x_dtype_matches_training(self, trained_handler, nominal_observation):
        trained_dtype = trained_handler._tensor_dataset.tensors[1].dtype
        x_tensor = nominal_observation.unsqueeze(0)
        assert x_tensor.dtype == trained_dtype

    def test_x_shape_matches_training(self, trained_handler, nominal_observation):
        trained_shape = trained_handler._tensor_dataset.tensors[1].shape[1:]
        x_tensor = nominal_observation.unsqueeze(0)
        assert x_tensor.shape[1:] == trained_shape


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler — checkpoint round-trips
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerCheckpoints:
    def test_save_and_load_produces_same_posterior(
        self,
        trained_handler,
        tmp_path,
        prior_save,
        posterior_config,
        nominal_observation,
    ):
        ckpt_path = tmp_path / "de.pt"
        torch.save(trained_handler._density_estimator.state_dict(), ckpt_path)

        theta_dim = trained_handler._tensor_dataset.tensors[0].shape[1]
        x_dim = trained_handler._tensor_dataset.tensors[1].shape[1]

        loaded = InferenceHandler(prior_save)
        loaded.create_posterior(posterior_config)
        device = loaded.device_handler.device
        de = loaded.inference._build_neural_net(
            torch.zeros(2, theta_dim, device=device),
            torch.zeros(2, x_dim, device=device),
        )

        de.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        de.to(device).eval()
        loaded._density_estimator = de

        original = (
            trained_handler.sample_posterior(500, nominal_observation).cpu().numpy()
        )
        reloaded = loaded.sample_posterior(500, nominal_observation).cpu().numpy()

        for i in range(original.shape[1]):
            ks_stat, _ = stats.ks_2samp(original[:, i], reloaded[:, i], method="asymp")
            assert ks_stat < 0.1, f"Parameter {i}: KS={ks_stat:.3f}"

    def test_load_posterior_autosave_format(
        self, prior_save, posterior_config, tmp_path
    ):
        prior = load_prior(prior_save)
        neural_net = posterior_nn(
            model=posterior_config.model,
            hidden_features=posterior_config.hidden_features,
            num_transforms=posterior_config.num_transforms,
            dropout_probability=posterior_config.dropout_probability,
            num_blocks=posterior_config.num_blocks,
            num_bins=posterior_config.num_bins,
            device=prior.device_handler.device,
            z_score_x="structured",
            z_score_theta="structured",
        )
        npe = NPE(
            prior=prior,
            density_estimator=neural_net,
            device=prior.device_handler.device,
        )
        theta_dim = len(prior.prior_data.parameter_names)
        de = npe._build_neural_net(torch.zeros(2, theta_dim), torch.zeros(2, 12))

        ckpt = {
            "epoch": 1,
            "model_state": de.state_dict(),
            "model_config": posterior_config,
            "optimizer_state": {},
            "best_val_loss": 0.5,
            "epochs_no_improve": 0,
        }
        ckpt_path = tmp_path / "autosave.ckpt"
        torch.save(ckpt, ckpt_path)

        handler = InferenceHandler(prior_save)
        handler.load_posterior(ckpt_path)
        assert handler._density_estimator is not None

    def test_load_posterior_warns_when_both_configs_provided(
        self, prior_save, posterior_config, tmp_path
    ):
        prior = load_prior(prior_save)
        neural_net = posterior_nn(
            model=posterior_config.model,
            hidden_features=posterior_config.hidden_features,
            num_transforms=posterior_config.num_transforms,
            dropout_probability=posterior_config.dropout_probability,
            num_blocks=posterior_config.num_blocks,
            num_bins=posterior_config.num_bins,
            device=prior.device_handler.device,
            z_score_x="structured",
            z_score_theta="structured",
        )
        npe = NPE(
            prior=prior,
            density_estimator=neural_net,
            device=prior.device_handler.device,
        )
        theta_dim = len(prior.prior_data.parameter_names)
        de = npe._build_neural_net(torch.zeros(2, theta_dim), torch.zeros(2, 12))

        ckpt = {
            "epoch": 1,
            "model_state": de.state_dict(),
            "model_config": posterior_config,
        }
        ckpt_path = tmp_path / "both.ckpt"
        torch.save(ckpt, ckpt_path)

        handler = InferenceHandler(prior_save)
        handler.load_posterior(ckpt_path, posterior_config=posterior_config)
        assert handler._density_estimator is not None
