from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.simulator import load_prior
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig
from mach3sbitools.utils.device_handler import TorchDeviceHandler
from mach3sbitools.utils.logger import get_logger

from .training import SBITrainer

logger = get_logger()


class InferenceHandler:
    """
    High-level interface combining simulation, data loading,
    and NPE training with a custom high-performance training loop.
    """

    def __init__(
        self,
        prior_path: Path,
        nuisance_pars: list[str] | None = None,
    ):
        self.nuisance_pars = nuisance_pars

        self.prior = load_prior(prior_path)
        # Might as well grab this from the prior
        self.parameter_names = self.prior.prior_data.parameter_names

        if nuisance_pars is not None:
            self.prior.set_nuisance_filter(nuisance_pars)

        self.dataset: ParaketDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None

        self._density_estimator: nn.Module | None = None
        self._tensor_dataset: TensorDataset | None = None
        self.device_handler = TorchDeviceHandler()

    # ── Data ─────────────────────────────────────────────────────────────────

    def set_dataset(self, data_folder: Path) -> None:
        """Point the interface at a folder of feather files."""
        self.dataset = ParaketDataset(data_folder, self.parameter_names, self.nuisance_pars)
        logger.info(
            f"Dataset set: [bold]{len(self.dataset)}[/] files in [cyan]{data_folder}[/]"
        )

    def load_training_data(self) -> None:
        """
        Pre-load all feather files into RAM as a flat TensorDataset.
        Call once before training — avoids repeated disk reads per epoch.
        Data is kept on CPU; pinned-memory DataLoader handles GPU transfers.
        """
        if self.dataset is None:
            raise ValueError("Call set_dataset() before load_training_data().")

        # Build nuisance mask if needed
        self._tensor_dataset = self.dataset.to_tensor_dataset(
            device="cpu",
        )

    # ── Model ─────────────────────────────────────────────────────────────────

    def create_posterior(
        self,
        config: PosteriorConfig,
    ) -> None:
        """
        Build the NPE inference object and density estimator.

        Args:
            config: PosteriorConfig object containing model parameters.
        """
        neural_net = posterior_nn(
            model=config.model,
            hidden_features=config.hidden_features,
            num_transforms=config.num_transforms,
            dropout_probability=config.dropout_probability,
            num_blocks=config.num_blocks,
            num_bins=config.num_bins,
        )

        self.inference = NPE(
            prior=self.prior,
            density_estimator=neural_net,
            device=self.device_handler.device,
        )

        logger.info(
            f"NPE created | {config.model} | "
            f"hidden=[cyan]{config.hidden_features}[/] transforms=[cyan]{config.num_transforms}[/] "
            f"blocks=[cyan]{config.num_blocks}[/] bins=[cyan]{config.num_bins}[/]"
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train_posterior(
        self,
        config: TrainingConfig,
        # resume_checkpoint: Optional[Path] = None,
    ) -> None:
        """
        Train the posterior density estimator using the custom training loop.

        Args:
            config:            TrainingConfig object containing all training parameters.
            resume_checkpoint: Path to a checkpoint file produced by SBITrainer.save_checkpoint().
                               All training state (weights, optimizer, scheduler, scaler,
                               epoch counter, best val loss, early-stop counter) is restored
                               so training continues seamlessly from where it left off.
                               When provided, create_posterior() does not need to have been
                               called first — the density estimator is rebuilt automatically
                               from the data dimensions and then loaded from the checkpoint.
        """
        if self._tensor_dataset is None:
            raise ValueError("Call load_training_data() before train_posterior().")

        # If resuming without a live inference object, we need to reconstruct it.
        if config.resume_checkpoint is not None and self.inference is None:
            raise ValueError(
                "Call create_posterior() before train_posterior() so the network "
                "architecture is defined. Weights will be overwritten by the checkpoint."
            )

        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

        # Build the raw density estimator network from sbi,
        # using a small sample to infer theta/x dimensions.
        sample_theta = self._tensor_dataset.tensors[0][:10]
        sample_x = self._tensor_dataset.tensors[1][:10]
        density_estimator = self.inference._build_neural_net(sample_theta, sample_x)

        trainer = SBITrainer(
            dataset=self._tensor_dataset,
            config=config,
            device=self.device_handler.device,
        )

        self._density_estimator = trainer.train(
            density_estimator,
            config,
            resume_checkpoint=config.resume_checkpoint,
        )

    # ── Posterior sampling ────────────────────────────────────────────────────

    def build_posterior(self) -> None:
        if self._density_estimator is None:
            raise ValueError("Train or load a density estimator first.")

        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

        self.posterior = self.inference.build_posterior(self._density_estimator)

    def sample_posterior(
        self,
        num_samples: int,
        x: list[float],
        **kwargs,
    ) -> torch.Tensor:
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()

        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")

        x_tensor = torch.tensor(
            [x], dtype=torch.float32, device=self.device_handler.device
        )
        return cast(torch.Tensor, self.posterior.sample((num_samples,), x=x_tensor, **kwargs))

    def load_posterior(
        self,
        checkpoint_path: Path,
        config: PosteriorConfig,
    ) -> None:
        """
        Load a trained density estimator directly into the interface,
        ready for sample_posterior(). Dimensions are inferred from the
        simulator's prior and data bins.

        Args:
            checkpoint_path: Path to best-model or autosave checkpoint.
            config:          PosteriorConfig matching the saved architecture.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            logger.info(f"Loading autosave checkpoint from epoch {ckpt['epoch']}")
        else:
            state_dict = ckpt
            logger.info("Loading best-model state dict")

        self.create_posterior(config)

        theta_dim = self.prior.n_params
        x_dim = self.prior.event_shape[0]

        if self.inference is None:
            raise ValueError("Cannot find inference.")

        # Need to build a dummy first
        density_estimator = self.inference._build_neural_net(
            torch.zeros(2, theta_dim),
            torch.zeros(2, x_dim),
        )

        density_estimator.load_state_dict(state_dict)
        density_estimator.to(self.device_handler.device).eval()
        self._density_estimator = density_estimator

        logger.info(f"Density estimator loaded from [cyan]{checkpoint_path}[/]")
