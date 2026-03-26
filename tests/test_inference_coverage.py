"""
Coverage-boosting tests for:
  - inference/inference_handler.py  (lines 73, 162, 165, 199, 201, 225, 259-335, 347-363)
  - inference/training.py           (lines 82, 155-159, 225, 279-280, 286, 309, 311,
                                     372, 380-411, 417, 437, 450, 463-466, 482, 484,
                                     486, 505, 528-533)
"""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from mach3sbitools.exceptions import (
    DensityEstimatorError,
    OptimizerNotSpecified,
    SBITrainingException,
    ScalarNotSpecified,
)
from mach3sbitools.inference import InferenceHandler
from mach3sbitools.inference.training import SBITrainer, save_checkpoint
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tiny_dataset(n: int = 200, theta_dim: int = 4, x_dim: int = 6) -> TensorDataset:
    theta = torch.randn(n, theta_dim)
    x = torch.randn(n, x_dim)
    return TensorDataset(theta, x)


def _tiny_model(theta_dim: int = 4, x_dim: int = 6) -> torch.nn.Module:
    """A tiny linear network that exposes a .loss() method for the trainer."""

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(x_dim, theta_dim)

        def forward(self, x):
            return self.fc(x)

        def loss(self, theta, x):
            pred = self.fc(x)
            return ((pred - theta) ** 2).mean(dim=-1)

    return TinyModel()


def _minimal_config(tmp_path: Path, **kwargs) -> TrainingConfig:
    defaults: dict[str, object] = dict(
        save_path=tmp_path / "model.ckpt",
        batch_size=32,
        max_epochs=2,
        stop_after_epochs=10,
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
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# save_checkpoint
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSaveCheckpoint:
    def test_saves_checkpoint_file(self, tmp_path):
        model = _tiny_model()
        opt = torch.optim.Adam(model.parameters())
        from torch.amp import GradScaler
        from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

        warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=5)
        plateau = ReduceLROnPlateau(opt)
        scaler = GradScaler(device="cpu", enabled=False)

        save_checkpoint(
            epoch=1,
            density_estimator=model,
            optimizer=opt,
            warmup_scheduler=warmup,
            plateau_scheduler=plateau,
            scaler=scaler,
            best_val_loss=0.5,
            epochs_no_improve=0,
            save_path=tmp_path / "run.ckpt",
            model_config=PosteriorConfig(),
            use_unique_path=True,
        )

        ckpt_files = list((tmp_path / "checkpoints").glob("*.ckpt"))
        assert len(ckpt_files) == 1

    def test_saves_at_exact_path_when_not_unique(self, tmp_path):
        model = _tiny_model()
        opt = torch.optim.Adam(model.parameters())
        from torch.amp import GradScaler
        from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

        warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=5)
        plateau = ReduceLROnPlateau(opt)
        scaler = GradScaler(device="cpu", enabled=False)
        target = tmp_path / "exact.ckpt"

        save_checkpoint(
            epoch=1,
            density_estimator=model,
            optimizer=opt,
            warmup_scheduler=warmup,
            plateau_scheduler=plateau,
            scaler=scaler,
            best_val_loss=0.5,
            epochs_no_improve=0,
            save_path=target,
            use_unique_path=False,
        )

        assert target.exists()


# ─────────────────────────────────────────────────────────────────────────────
# SBITrainer error paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSBITrainerErrors:
    def test_raises_density_estimator_error_when_none(self, tmp_path):
        """Covers line 225 — DensityEstimatorError when model is None."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        with pytest.raises(DensityEstimatorError):
            trainer.train(None)  # type: ignore[arg-type]

    def test_train_one_epoch_raises_without_optimizer(self, tmp_path):
        """Covers OptimizerNotSpecified in _train_one_epoch (line 309)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.scaler = MagicMock()
        # optimizer is None by default before train()
        with pytest.raises(OptimizerNotSpecified):
            trainer._train_one_epoch(_tiny_model(), nullcontext())

    def test_train_one_epoch_raises_without_scaler(self, tmp_path):
        """Covers ScalarNotSpecified in _train_one_epoch (line 311)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.optimizer = torch.optim.Adam(_tiny_model().parameters())
        # scaler is None by default before train()
        with pytest.raises(ScalarNotSpecified):
            trainer._train_one_epoch(_tiny_model(), nullcontext())

    def test_log_epoch_raises_without_optimizer(self, tmp_path):
        """Covers OptimizerNotSpecified in _log_epoch (line 372)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        with pytest.raises(OptimizerNotSpecified):
            trainer._log_epoch(1, 1.0, 1.0, 1.0)

    def test_step_schedulers_raises_without_schedulers(self, tmp_path):
        """Covers SBITrainingException in _step_schedulers (line 437)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.optimizer = torch.optim.Adam(_tiny_model().parameters())
        # warmup/plateau are None before train()
        with pytest.raises(SBITrainingException):
            trainer._step_schedulers(1)

    def test_maybe_save_raises_without_optimizer(self, tmp_path):
        """Covers OptimizerNotSpecified in _maybe_save (line 450)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        with pytest.raises(OptimizerNotSpecified):
            trainer._maybe_save(1, _tiny_model())

    def test_maybe_save_raises_without_schedulers(self, tmp_path):
        """Covers SBITrainingException in _maybe_save."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.optimizer = torch.optim.Adam(_tiny_model().parameters())
        with pytest.raises(SBITrainingException):
            trainer._maybe_save(1, _tiny_model())

    def test_resume_raises_without_optimizer(self, tmp_path):
        """Covers OptimizerNotSpecified in _resume_if_requested (line 417)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        # Pass a dummy path — the check for optimizer happens before the load
        fake_ckpt = tmp_path / "fake.ckpt"
        fake_ckpt.touch()
        with pytest.raises(OptimizerNotSpecified):
            trainer._resume_if_requested(_tiny_model(), fake_ckpt)

    def test_build_schedulers_raises_without_optimizer(self, tmp_path):
        """Covers OptimizerNotSpecified in _build_schedulers (line 463)."""
        cfg = _minimal_config(tmp_path)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        with pytest.raises(OptimizerNotSpecified):
            trainer._build_schedulers()


@pytest.mark.slow
class TestSBITrainerCompile:
    def test_compile_false_returns_model_unchanged(self, tmp_path):
        """Covers _maybe_compile with compile=False (line 528)."""
        cfg = _minimal_config(tmp_path, compile=False)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        result = trainer._maybe_compile(model)
        assert result is model

    def test_compile_true_falls_back_gracefully(self, tmp_path):
        """Covers _maybe_compile with compile=True — torch.compile may or may not succeed."""
        cfg = _minimal_config(tmp_path, compile=True)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        # Should not raise — either compiles or logs a warning and returns model
        result = trainer._maybe_compile(model)
        assert result is not None


class TestSBITrainerTensorboard:
    def test_tensorboard_writer_initialised_when_dir_given(self, tmp_path):
        tb_dir = tmp_path / "tb"
        cfg = _minimal_config(tmp_path, tensorboard_dir=tb_dir)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        assert trainer.writer is not None

    @pytest.mark.slow
    def test_log_tensorboard_skips_when_no_writer(self, tmp_path):
        cfg = _minimal_config(tmp_path, tensorboard_dir=None)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.optimizer = torch.optim.Adam(_tiny_model().parameters())
        # Should silently return without raising
        trainer._log_tensorboard(1, 1.0, 1.0, 1.0)

    def test_log_tensorboard_raises_without_optimizer_when_writer_present(
        self, tmp_path
    ):
        """Covers OptimizerNotSpecified in _log_tensorboard (line 482)."""
        tb_dir = tmp_path / "tb"
        cfg = _minimal_config(tmp_path, tensorboard_dir=tb_dir)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.writer = MagicMock()
        with pytest.raises(OptimizerNotSpecified):
            trainer._log_tensorboard(1, 1.0, 1.0, 1.0)


@pytest.mark.slow
class TestSBITrainerFullRun:
    def test_trains_for_two_epochs(self, tmp_path):
        """End-to-end smoke test: 2 epochs, no tensorboard, no AMP."""
        cfg = _minimal_config(tmp_path, max_epochs=2, autosave_every=2)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        result = trainer.train(model)
        assert isinstance(result, torch.nn.Module)

    def test_best_state_is_restored(self, tmp_path):
        """After training the returned model should have a state dict."""
        cfg = _minimal_config(tmp_path, max_epochs=2)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        result = trainer.train(model)
        assert len(result.state_dict()) > 0

    def test_checkpoint_written(self, tmp_path):
        """Periodic checkpoint should appear in the checkpoints/ dir."""
        cfg = _minimal_config(tmp_path, max_epochs=2, autosave_every=1)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        trainer.train(_tiny_model())
        ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
        assert len(ckpts) > 0

    def test_resume_from_checkpoint(self, tmp_path):
        """Covers _resume_if_requested happy path (lines 380-411)."""
        # First run to produce a checkpoint
        cfg1 = _minimal_config(tmp_path, max_epochs=2, autosave_every=1)
        trainer1 = SBITrainer(_tiny_dataset(), cfg1, device="cpu")
        trainer1.train(_tiny_model())
        ckpt = sorted((tmp_path / "checkpoints").glob("*.ckpt"))[0]

        # Second run resumes
        cfg2 = _minimal_config(
            tmp_path, max_epochs=4, autosave_every=100, resume_checkpoint=ckpt
        )
        trainer2 = SBITrainer(_tiny_dataset(), cfg2, device="cpu")
        model2 = _tiny_model()
        result = trainer2.train(model2, resume_checkpoint=ckpt)
        assert result is not None

    def test_early_stopping_fires(self, tmp_path):
        """Covers early-stopping branch (line 286)."""
        cfg = _minimal_config(
            tmp_path, max_epochs=100, stop_after_epochs=1, autosave_every=200
        )
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        result = trainer.train(_tiny_model())
        assert result is not None

    def test_step_schedulers_warmup_path(self, tmp_path):
        """Covers the warmup branch in _step_schedulers (epoch <= warmup_epochs)."""
        cfg = _minimal_config(tmp_path, warmup_epochs=10)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        trainer.optimizer = torch.optim.Adam(model.parameters())
        trainer.warmup, trainer.plateau = trainer._build_schedulers()
        # epoch=5 is within warmup window
        trainer._step_schedulers(epoch=5)

    def test_step_schedulers_plateau_path(self, tmp_path):
        """Covers the plateau branch in _step_schedulers (epoch > warmup_epochs)."""
        cfg = _minimal_config(tmp_path, warmup_epochs=2)
        trainer = SBITrainer(_tiny_dataset(), cfg, device="cpu")
        model = _tiny_model()
        trainer.optimizer = torch.optim.Adam(model.parameters())
        trainer.warmup, trainer.plateau = trainer._build_schedulers()
        trainer.ema_val_loss = 1.0
        # epoch=5 is beyond warmup window
        trainer._step_schedulers(epoch=5)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler error paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerErrors:
    def test_load_posterior_raises_file_not_found(self, prior_save):
        """Covers FileNotFoundError in load_posterior (line 259)."""
        handler = InferenceHandler(prior_save)
        with pytest.raises(FileNotFoundError):
            handler.load_posterior(Path("/no/such/checkpoint.pt"))

    def test_build_posterior_raises_without_density_estimator(self, prior_save):
        """Covers ValueError in build_posterior (line 199)."""
        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError, match="density estimator"):
            handler.build_posterior()

    def test_build_posterior_raises_without_inference(self, prior_save):
        """Covers ValueError in build_posterior when inference is None (line 201)."""
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()  # set de but not inference
        with pytest.raises(ValueError):
            handler.build_posterior()

    def test_sample_posterior_raises_without_estimator(self, prior_save):
        """Covers ValueError in sample_posterior (line 225)."""
        handler = InferenceHandler(prior_save)
        # Patch build_posterior to set posterior=None
        handler._density_estimator = MagicMock()
        handler.inference = MagicMock()
        handler.posterior = None
        with patch.object(handler, "build_posterior"):
            with pytest.raises(ValueError, match="density estimator"):
                handler.sample_posterior(10, [1.0] * 12)

    def test_train_posterior_raises_if_no_tensor_dataset(
        self, prior_save, dummy_data_dir, posterior_config, training_config
    ):
        """Covers line 162 — ValueError when _tensor_dataset is None."""
        handler = InferenceHandler(prior_save)
        handler.create_posterior(posterior_config)
        with pytest.raises(ValueError, match="load_training_data"):
            handler.train_posterior(training_config)

    def test_train_posterior_raises_if_no_inference(
        self, prior_save, dummy_data_dir, training_config
    ):
        """Covers line 165 — ValueError when inference is None but resume_checkpoint given."""
        from torch.utils.data import TensorDataset

        handler = InferenceHandler(prior_save)
        handler._tensor_dataset = TensorDataset(torch.zeros(10, 4), torch.zeros(10, 6))
        # inference is None, resume_checkpoint is set
        cfg = TrainingConfig(resume_checkpoint=Path("/some/ckpt.pt"))
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(cfg)

    def test_load_posterior_autosave_format(
        self, prior_save, posterior_config, tmp_path
    ):
        """Covers the 'model_state' checkpoint branch in load_posterior."""
        # Build a tiny model and fake checkpoint
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn

        from mach3sbitools.simulator import load_prior

        prior = load_prior(prior_save)
        neural_net = posterior_nn(
            model=posterior_config.model,
            hidden_features=posterior_config.hidden_features,
            num_transforms=posterior_config.num_transforms,
            dropout_probability=posterior_config.dropout_probability,
            num_blocks=posterior_config.num_blocks,
            num_bins=posterior_config.num_bins,
        )
        npe = NPE(prior=prior, density_estimator=neural_net)

        theta_dim = len(prior.prior_data.parameter_names)
        x_dim = 12
        de = npe._build_neural_net(torch.zeros(2, theta_dim), torch.zeros(2, x_dim))

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
        """Covers the warning logged when both checkpoint config and caller config exist."""
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn

        from mach3sbitools.simulator import load_prior

        prior = load_prior(prior_save)
        neural_net = posterior_nn(
            model=posterior_config.model,
            hidden_features=posterior_config.hidden_features,
            num_transforms=posterior_config.num_transforms,
            dropout_probability=posterior_config.dropout_probability,
            num_blocks=posterior_config.num_blocks,
            num_bins=posterior_config.num_bins,
        )
        npe = NPE(prior=prior, density_estimator=neural_net)
        theta_dim = len(prior.prior_data.parameter_names)
        x_dim = 12
        de = npe._build_neural_net(torch.zeros(2, theta_dim), torch.zeros(2, x_dim))

        ckpt = {
            "epoch": 1,
            "model_state": de.state_dict(),
            "model_config": posterior_config,
        }
        ckpt_path = tmp_path / "both.ckpt"
        torch.save(ckpt, ckpt_path)

        handler = InferenceHandler(prior_save)
        # Passing posterior_config alongside a checkpoint that also has one → warning
        handler.load_posterior(ckpt_path, posterior_config=posterior_config)
        assert handler._density_estimator is not None

    def test_load_posterior_plain_state_dict_raises_without_config(
        self, prior_save, tmp_path
    ):
        """Covers the plain-state-dict branch raising when no config given."""
        fake_state = {"some_key": torch.zeros(1)}
        ckpt_path = tmp_path / "plain.pt"
        torch.save(fake_state, ckpt_path)

        handler = InferenceHandler(prior_save)
        with pytest.raises(ValueError, match="model_config"):
            handler.load_posterior(ckpt_path, posterior_config=None)

    def test_get_log_likelihood_raises_without_estimator(self, prior_save):
        """Covers ValueError in get_log_likelihood (line 347)."""
        handler = InferenceHandler(prior_save)
        handler._density_estimator = MagicMock()
        handler.inference = MagicMock()
        handler.posterior = None
        with patch.object(handler, "build_posterior"):
            with pytest.raises(ValueError, match="density estimator"):
                handler.get_log_likelihood(np.ones((5, 4)), [1.0] * 12)
