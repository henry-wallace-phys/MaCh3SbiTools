import fnmatch
import pickle as pkl
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from mach3sbitools.mach3_interface.mach3_simulator import MaCh3Simulator
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.utils.device_handler import TorchDeviceHandler
from mach3sbitools.inference.sbi_trainer import SBITrainer
from mach3sbitools.utils.config import TrainingConfig, PosteriorConfig
from mach3sbitools.utils.logger import get_logger

logger = get_logger(__name__)


class MaCh3SBIInterface:
    """
    High-level interface combining MaCh3 simulation, data loading,
    and NPE training with a custom high-performance training loop.
    """

    def __init__(
        self,
        mach3_name: str,
        config_file: Path,
        nuisance_pars: Optional[List[str]] = None,
        cyclical_pars: List[str] | None = None,
    ):
        self.simulator = MaCh3Simulator(mach3_name, config_file, nuisance_pars, cyclical_pars)
        self.nuisance_pars = nuisance_pars
        self.mach3_name = mach3_name

        self.dataset: Optional[ParaketDataset] = None
        self.inference: Optional[NPE] = None
        self.posterior = None

        self._density_estimator: Optional[nn.Module] = None
        self._tensor_dataset: Optional[TensorDataset] = None
        self._nuisance_mask: Optional[torch.BoolTensor] = None

        self.device_handler = TorchDeviceHandler()

    # ── Data ─────────────────────────────────────────────────────────────────

    def set_dataset(self, data_folder: Path) -> None:
        """Point the interface at a folder of feather files."""
        self.dataset = ParaketDataset(data_folder)
        logger.info(f"Dataset set: [bold]{len(self.dataset)}[/] files in [cyan]{data_folder}[/]")

    def load_training_data(self) -> None:
        """
        Pre-load all feather files into RAM as a flat TensorDataset.
        Call once before training — avoids repeated disk reads per epoch.
        Data is kept on CPU; pinned-memory DataLoader handles GPU transfers.
        """
        if self.dataset is None:
            raise ValueError("Call set_dataset() before load_training_data().")

        # Build nuisance mask if needed
        parameter_names = self.simulator.mach3_wrapper.get_parameter_names()
        if self.nuisance_pars is not None:
            mask = [
                not any(fnmatch.fnmatch(name, n) for n in self.nuisance_pars)
                for name in parameter_names
            ]
            self._nuisance_mask = torch.tensor(mask, dtype=torch.bool)
        else:
            self._nuisance_mask = None

        self._tensor_dataset = self.dataset.to_tensor_dataset(
            device="cpu",
            nuisance_mask=self._nuisance_mask,
        )

    # ── Model ─────────────────────────────────────────────────────────────────

    def create_posterior(
        self,
        config: PosteriorConfig,
    ) -> None:
        """
        Build the NPE inference object and NSF density estimator.

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
            prior=self.simulator.prior,
            density_estimator=neural_net,
            device=self.device_handler.device,
        )

        logger.info(
            f"NPE created | NSF | "
            f"hidden=[cyan]{config.hidden_features}[/] transforms=[cyan]{config.num_transforms}[/] "
            f"blocks=[cyan]{config.num_blocks}[/] bins=[cyan]{config.num_bins}[/]"
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train_posterior(
        self,
        config: TrainingConfig,
    ) -> None:
        """
        Train the posterior density estimator using the custom training loop.

        Args:
            config: TrainingConfig object containing all training parameters.
        """
        if self._tensor_dataset is None:
            raise ValueError("Call load_training_data() before train_posterior().")
        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

        save_file = config.save_path or Path(f"{self.mach3_name}_best.pt")

        # Build the raw density estimator network from sbi,
        # using a small sample to infer theta/x dimensions
        sample_theta = self._tensor_dataset.tensors[0][:100]
        sample_x = self._tensor_dataset.tensors[1][:100]
        density_estimator = self.inference._build_neural_net(sample_theta, sample_x)

        trainer = SBITrainer(
            dataset=self._tensor_dataset,
            config=config,
            device=self.device_handler.device,
        )

        self._density_estimator = trainer.train(
            density_estimator,
            config,
        )

    # ── Posterior sampling ────────────────────────────────────────────────────

    def build_posterior(self) -> None:
        if self._density_estimator is None:
            raise ValueError("Train or load a density estimator first.")
        self.posterior = self.inference.build_posterior(self._density_estimator)

    def sample_posterior(
        self,
        num_samples: int = 1000,
        x: Optional[List[float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()

        if x is None:
            x = self.simulator.mach3_wrapper.get_data_bins()

        x_tensor = torch.tensor(
            [x], dtype=torch.float32, device=self.device_handler.device
        )
        return self.posterior.sample((num_samples,), x=x_tensor, **kwargs)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_density_estimator(self, file_path: Path) -> None:
        """Save just the trained network weights."""
        if self._density_estimator is None:
            raise ValueError("No density estimator to save.")
        torch.save(self._density_estimator.state_dict(), file_path)
        logger.info(f"Density estimator saved → [cyan]{file_path}[/]")

    def load_density_estimator(self, file_path: Path) -> None:
        """Load weights into an already-created density estimator."""
        if self._density_estimator is None:
            raise ValueError("Call create_posterior() before loading weights.")
        state = torch.load(file_path, map_location=self.device_handler.device)
        self._density_estimator.load_state_dict(state)
        self._density_estimator.to(self.device_handler.device)
        logger.info(f"Density estimator loaded ← [cyan]{file_path}[/]")

    def save_inference(self, file_path: Path) -> None:
        """Pickle the full sbi NPE object (for compatibility)."""
        if self.inference is None:
            raise ValueError("No inference object to save.")
        with open(file_path, "wb") as f:
            pkl.dump(self.inference, f)

    def load_inference(self, file_path: Path) -> None:
        """Load a pickled sbi NPE object."""
        with open(file_path, "rb") as f:
            self.inference = pkl.load(f)