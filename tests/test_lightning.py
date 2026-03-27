"""
Tests for the PyTorch Lightning training components.

Covers:
  - inference/lightning_module.py
  - inference/lightning_datamodule.py
  - InferenceHandler.train_posterior (Lightning path)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from mach3sbitools.data_loaders import SBIDataModule
from mach3sbitools.inference import InferenceHandler
from mach3sbitools.inference.lightning_module import SBILightningModule
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tiny_dataset(n: int = 200, theta_dim: int = 4, x_dim: int = 6) -> TensorDataset:
    theta = torch.randn(n, theta_dim)
    x = torch.randn(n, x_dim)
    return TensorDataset(theta, x)


def _tiny_model(theta_dim: int = 4, x_dim: int = 6) -> torch.nn.Module:
    """Minimal network that exposes a .loss() method matching the sbi interface."""

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
        batch_size=32,
        max_epochs=2,
        stop_after_epochs=10,
        autosave_every=1,
        print_interval=1,
        show_progress=False,
        validation_fraction=0.1,
        num_workers=0,
        ema_alpha=0.1,
        scheduler_patience=5,
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# SBILightningModule — unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSBILightningModuleInit:
    def test_stores_model(self, tmp_path):
        model = _tiny_model()
        cfg = _minimal_config(tmp_path)
        module = SBILightningModule(model, cfg)
        assert module.model is model

    def test_stores_config(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        module = SBILightningModule(_tiny_model(), cfg)
        assert module.config is cfg

    def test_stores_model_config(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        pc = PosteriorConfig()
        module = SBILightningModule(_tiny_model(), cfg, model_config=pc)
        assert module.model_config is pc

    def test_ema_val_loss_initialised_to_inf(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        assert module.ema_val_loss == float("inf")


class TestSBILightningModuleForward:
    def test_forward_returns_tensor(self, tmp_path):
        model = _tiny_model()
        module = SBILightningModule(model, _minimal_config(tmp_path))
        theta = torch.randn(8, 4)
        x = torch.randn(8, 6)
        out = module(theta, x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8,)


class TestSBILightningModuleTrainingStep:
    def test_returns_scalar_loss(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        # Provide a mock trainer so self.log doesn't error
        module.trainer = MagicMock()
        module._log_dict = {}

        with patch.object(module, "log"):
            loss = module.training_step((torch.randn(8, 4), torch.randn(8, 6)), 0)

        assert loss.ndim == 0  # scalar

    def test_calls_log(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch.object(module, "log") as mock_log:
            module.training_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "train_loss"


class TestSBILightningModuleValidationStep:
    def test_returns_scalar_loss(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch.object(module, "log"):
            loss = module.validation_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        assert loss.ndim == 0

    def test_calls_log(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch.object(module, "log") as mock_log:
            module.validation_step((torch.randn(8, 4), torch.randn(8, 6)), 0)
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "val_loss"


class TestSBILightningModuleEMA:
    def test_ema_updates_from_inf(self, tmp_path):
        """First epoch: EMA should equal val_loss exactly."""
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val_loss": torch.tensor(2.0)}

        with patch.object(module, "log"):
            module.on_validation_epoch_end()

        assert module.ema_val_loss == pytest.approx(2.0)

    def test_ema_smooths_on_subsequent_epochs(self, tmp_path):
        """Second epoch: EMA should be a weighted average."""
        cfg = _minimal_config(tmp_path)
        cfg.ema_alpha = 0.1
        module = SBILightningModule(_tiny_model(), cfg)
        module.ema_val_loss = 2.0
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val_loss": torch.tensor(1.0)}

        with patch.object(module, "log"):
            module.on_validation_epoch_end()

        # 0.1 * 1.0 + 0.9 * 2.0 = 1.9
        assert module.ema_val_loss == pytest.approx(1.9)

    def test_ema_logged(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        module.trainer = MagicMock()
        module.trainer.callback_metrics = {"val_loss": torch.tensor(1.5)}

        with patch.object(module, "log") as mock_log:
            module.on_validation_epoch_end()

        logged_keys = [c[0][0] for c in mock_log.call_args_list]
        assert "ema_val_loss" in logged_keys


class TestSBILightningModuleGPULogging:
    def test_gpu_logging_skipped_when_no_cuda(self, tmp_path):
        """on_train_epoch_end should not raise when CUDA is unavailable."""
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(module, "log") as mock_log:
                module.on_train_epoch_end()
        mock_log.assert_not_called()

    @pytest.mark.slow
    def test_gpu_stats_logged_when_cuda_available(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024**2 * 100),
            patch("torch.cuda.memory_reserved", return_value=1024**2 * 200),
            patch.object(module, "log") as mock_log,
        ):
            module.on_train_epoch_end()

        logged_keys = [c[0][0] for c in mock_log.call_args_list]
        assert "gpu/allocated_mb" in logged_keys
        assert "gpu/reserved_mb" in logged_keys


class TestSBILightningModuleConfigureOptimizers:
    def test_returns_optimizer_and_scheduler(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_optimizer_is_adam(self, tmp_path):
        module = SBILightningModule(_tiny_model(), _minimal_config(tmp_path))
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.Adam)


# ─────────────────────────────────────────────────────────────────────────────
# SBIDataModule — unit tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSBIDataModuleInit:
    def test_stores_dataset_and_config(self, tmp_path):
        ds = _tiny_dataset()
        cfg = _minimal_config(tmp_path)
        dm = SBIDataModule(ds, cfg)
        assert dm.dataset is ds
        assert dm.config is cfg

    def test_train_val_datasets_none_before_setup(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        assert dm.train_dataset is None
        assert dm.val_dataset is None


class TestSBIDataModuleSetup:
    def test_setup_creates_splits(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(n=200), _minimal_config(tmp_path))
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_split_sizes_sum_to_dataset_length(self, tmp_path):
        n = 200
        cfg = _minimal_config(tmp_path)
        cfg.validation_fraction = 0.1
        dm = SBIDataModule(_tiny_dataset(n=n), cfg)
        dm.setup()
        assert len(dm.train_dataset) + len(dm.val_dataset) == n

    def test_val_fraction_respected(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        cfg.validation_fraction = 0.2
        dm = SBIDataModule(_tiny_dataset(n=100), cfg)
        dm.setup()
        assert len(dm.val_dataset) == 20

    def test_setup_is_deterministic(self, tmp_path):
        """Same seed should produce identical splits on repeated calls."""
        ds = _tiny_dataset(n=200)
        cfg = _minimal_config(tmp_path)

        dm1 = SBIDataModule(ds, cfg)
        dm1.setup()

        dm2 = SBIDataModule(ds, cfg)
        dm2.setup()

        # Compare the indices of the first few items
        assert dm1.train_dataset.indices[:10] == dm2.train_dataset.indices[:10]
        assert dm1.val_dataset.indices[:5] == dm2.val_dataset.indices[:5]

    def test_setup_accepts_stage_argument(self, tmp_path):
        """setup() must not raise when stage is passed (Lightning contract)."""
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup(stage="fit")
        assert dm.train_dataset is not None


class TestSBIDataModuleDataLoaders:
    def test_train_dataloader_returns_dataloader(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        assert isinstance(dm.train_dataloader(), DataLoader)

    def test_val_dataloader_returns_dataloader(self, tmp_path):

        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        assert isinstance(dm.val_dataloader(), DataLoader)

    def test_train_dataloader_batch_size(self, tmp_path):
        cfg = _minimal_config(tmp_path)
        cfg.batch_size = 16
        dm = SBIDataModule(_tiny_dataset(n=200), cfg)
        dm.setup()
        assert dm.train_dataloader().batch_size == 16

    def test_train_dataloader_shuffles(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        # DataLoader doesn't expose shuffle directly — check via sampler type
        assert isinstance(dm.train_dataloader().sampler, RandomSampler)

    def test_val_dataloader_does_not_shuffle(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        assert isinstance(dm.val_dataloader().sampler, SequentialSampler)

    def test_train_dataloader_drops_last(self, tmp_path):
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        assert dm.train_dataloader().drop_last is True

    def test_num_workers_is_zero(self, tmp_path):
        """Data is pre-loaded in RAM — workers should always be 0."""
        dm = SBIDataModule(_tiny_dataset(), _minimal_config(tmp_path))
        dm.setup()
        assert dm.train_dataloader().num_workers == 0
        assert dm.val_dataloader().num_workers == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end Lightning training smoke test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestLightningTrainingIntegration:
    def test_trainer_fits_without_error(self, tmp_path):
        """Full fit() call with a tiny model and dataset on CPU."""
        cfg = _minimal_config(tmp_path, max_epochs=2, stop_after_epochs=50)
        model = _tiny_model()
        module = SBILightningModule(model, cfg)
        dm = SBIDataModule(_tiny_dataset(n=200), cfg)

        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule=dm)

    def test_ema_val_loss_updated_after_training(self, tmp_path):
        cfg = _minimal_config(tmp_path, max_epochs=3, stop_after_epochs=50)
        module = SBILightningModule(_tiny_model(), cfg)
        dm = SBIDataModule(_tiny_dataset(n=200), cfg)

        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule=dm)
        assert module.ema_val_loss != float("inf")

    def test_checkpoint_written(self, tmp_path):

        cfg = _minimal_config(tmp_path, max_epochs=2, autosave_every=1)
        module = SBILightningModule(_tiny_model(), cfg)
        dm = SBIDataModule(_tiny_dataset(n=200), cfg)

        ckpt_cb = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            monitor="ema_val_loss",
            save_top_k=1,
            every_n_epochs=1,
        )
        trainer = L.Trainer(
            max_epochs=2,
            callbacks=[ckpt_cb],
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(module, datamodule=dm)
        assert len(list((tmp_path / "checkpoints").glob("*.ckpt"))) > 0


# ─────────────────────────────────────────────────────────────────────────────
# InferenceHandler.train_posterior — Lightning path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestInferenceHandlerLightning:
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
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(training_config)

    def test_train_posterior_raises_resume_without_inference(
        self, prior_save, tmp_path
    ):

        handler = InferenceHandler(prior_save)
        handler._tensor_dataset = TensorDataset(torch.zeros(10, 4), torch.zeros(10, 6))
        cfg = TrainingConfig(resume_checkpoint=Path("/some/ckpt.pt"))
        with pytest.raises(ValueError, match="create_posterior"):
            handler.train_posterior(cfg)

    def test_train_posterior_sets_density_estimator(
        self, prior_save, dummy_data_dir, posterior_config, tmp_path
    ):
        """After training, _density_estimator must be set and in eval mode."""

        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=2,
            stop_after_epochs=50,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
        )
        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        handler.create_posterior(posterior_config)
        handler.train_posterior(cfg, model_config=posterior_config)

        assert handler._density_estimator is not None
        assert not handler._density_estimator.training

    def test_slurm_nnodes_env_respected(
        self, prior_save, dummy_data_dir, posterior_config, tmp_path, monkeypatch
    ):
        """SLURM_NNODES env var should be passed through to the Trainer."""
        monkeypatch.setenv("SLURM_NNODES", "1")

        handler = InferenceHandler(prior_save)
        handler.set_dataset(dummy_data_dir)
        handler.load_training_data()
        handler.create_posterior(posterior_config)

        cfg = TrainingConfig(
            save_path=tmp_path / "model.ckpt",
            max_epochs=1,
            batch_size=256,
            show_progress=False,
            autosave_every=500,
        )

        captured = {}

        original_init = L.Trainer.__init__

        def patched_init(self, *args, **kwargs):
            captured["num_nodes"] = kwargs.get("num_nodes")
            # Force CPU/single device to avoid NCCL issues in test
            kwargs["accelerator"] = "cpu"
            kwargs["devices"] = 1
            kwargs["strategy"] = "auto"
            original_init(self, *args, **kwargs)

        with patch.object(L.Trainer, "__init__", patched_init):
            handler.train_posterior(cfg)

        assert captured["num_nodes"] == 1
