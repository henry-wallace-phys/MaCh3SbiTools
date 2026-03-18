"""
High-level inference interface.

Combines data loading, NPE model construction, training, and posterior
sampling into a single object. The typical workflow is::

    handler = InferenceHandler(prior_path)
    handler.set_dataset(data_folder)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    handler.train_posterior(training_config)
    samples = handler.sample_posterior(10_000, x_observed)
"""

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from torch.utils.data import TensorDataset

from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.simulator import load_prior
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig
from mach3sbitools.utils.device_handler import TorchDeviceHandler
from mach3sbitools.utils.logger import get_logger

from .training import SBITrainer

logger = get_logger()


class InferenceHandler:
    """
    High-level interface for NPE training and posterior sampling.

    Manages the full inference pipeline: loading simulations from disk,
    building and training an NPE density estimator, and drawing posterior
    samples conditioned on observed data.
    """

    def __init__(
        self,
        prior_path: Path,
        nuisance_pars: list[str] | None = None,
    ):
        """
        Initialise the handler and load the prior.

        :param prior_path: Path to a pickled :class:`~mach3sbitools.simulator.Prior`
            file produced by :func:`~mach3sbitools.simulator.create_prior`.
        :param nuisance_pars: fnmatch patterns for parameters to exclude.
            Passed directly to :meth:`~mach3sbitools.simulator.Prior.set_nuisance_filter`.
        """
        self.prior = load_prior(prior_path)
        self.parameter_names = self.prior.prior_data.parameter_names
        self.nuisance_pars = nuisance_pars

        if nuisance_pars is not None:
            self.prior.set_nuisance_filter(nuisance_pars)

        self.dataset: ParaketDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None
        self._density_estimator: nn.Module | None = None
        self._tensor_dataset: TensorDataset | None = None
        self.device_handler = TorchDeviceHandler()

    # ── Data ──────────────────────────────────────────────────────────────────

    def set_dataset(self, data_folder: Path) -> None:
        """
        Point the handler at a folder of ``.feather`` simulation files.

        :param data_folder: Directory containing ``.feather`` files produced
            by :meth:`~mach3sbitools.simulator.Simulator.save`.
        """
        self.dataset = ParaketDataset(
            data_folder, self.parameter_names.tolist(), self.nuisance_pars
        )
        logger.info(
            f"Dataset set: [bold]{len(self.dataset)}[/] files in [cyan]{data_folder}[/]"
        )

    def load_training_data(self) -> None:
        """
        Pre-load all feather files into RAM as a flat :class:`~torch.utils.data.TensorDataset`.

        Call once before :meth:`train_posterior`. Keeps data on CPU; the
        DataLoader handles GPU transfers via pinned memory.

        :raises ValueError: If :meth:`set_dataset` has not been called.
        """
        if self.dataset is None:
            raise ValueError("Call set_dataset() before load_training_data().")

        self._tensor_dataset = self.dataset.to_tensor_dataset(device="cpu")

    # ── Model ─────────────────────────────────────────────────────────────────

    def create_posterior(self, config: PosteriorConfig) -> None:
        """
        Build the NPE inference object and density estimator network.

        :param config: Architecture and hyperparameter settings.
            See :class:`~mach3sbitools.utils.PosteriorConfig`.
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

    def train_posterior(self, config: TrainingConfig) -> None:
        """
        Train the density estimator using the custom :class:`~mach3sbitools.inference.SBITrainer`.

        :param config: Training loop settings.
            See :class:`~mach3sbitools.utils.TrainingConfig`.
        :raises ValueError: If :meth:`load_training_data` or
            :meth:`create_posterior` has not been called.
        """
        if self._tensor_dataset is None:
            raise ValueError("Call load_training_data() before train_posterior().")

        if config.resume_checkpoint is not None and self.inference is None:
            raise ValueError(
                "Call create_posterior() before train_posterior() so the network "
                "architecture is defined. Weights will be overwritten by the checkpoint."
            )

        if self.inference is None:
            raise ValueError("Call create_posterior() before train_posterior().")

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
            resume_checkpoint=config.resume_checkpoint,
        )

    # ── Posterior sampling ────────────────────────────────────────────────────

    def build_posterior(self) -> None:
        """
        Wrap the trained density estimator in an ``sbi`` posterior object.

        Called automatically by :meth:`sample_posterior`.

        :raises ValueError: If no density estimator has been trained or loaded.
        """
        if self._density_estimator is None:
            raise ValueError("Train or load a density estimator first.")
        if self.inference is None:
            raise ValueError("Call create_posterior() before build_posterior().")

        self.posterior = self.inference.build_posterior(self._density_estimator)

    def sample_posterior(
        self,
        num_samples: int,
        x: list[float],
        **kwargs,
    ) -> torch.Tensor:
        """
        Draw samples from the posterior conditioned on *x*.

        :param num_samples: Number of posterior samples to draw.
        :param x: Observed data vector *x_o*.
        :param kwargs: Additional keyword arguments forwarded to
            ``sbi.posterior.sample``.
        :returns: Tensor of shape ``(num_samples, n_params)``.
        :raises ValueError: If no density estimator is available.
        """
        logger.info(f"Sampling [bold]{num_samples:,}[/] points from posterior")
        self.build_posterior()

        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")

        x_tensor = torch.tensor(
            [x], dtype=torch.float32, device=self.device_handler.device
        )
        return cast(
            torch.Tensor, self.posterior.sample((num_samples,), x=x_tensor, **kwargs)
        )

    def load_posterior(
        self,
        checkpoint_path: Path,
        config: PosteriorConfig,
    ) -> None:
        """
        Load a trained density estimator from a checkpoint file.

        Supports both best-model state dicts (plain ``state_dict``) and
        autosave checkpoints (dicts with a ``"model_state"`` key).

        Parameter and observable dimensions are inferred from the prior.

        :param checkpoint_path: Path to a ``.pt`` checkpoint file.
        :param config: Architecture config — **must match** the settings used
            during training.
        :raises FileNotFoundError: If *checkpoint_path* does not exist.
        :raises ValueError: If the inference object is unavailable after
            :meth:`create_posterior`.
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

        density_estimator = self.inference._build_neural_net(
            torch.zeros(2, theta_dim),
            torch.zeros(2, x_dim),
        )
        density_estimator.load_state_dict(state_dict)
        density_estimator.to(self.device_handler.device).eval()
        self._density_estimator = density_estimator

        logger.info(f"Density estimator loaded from [cyan]{checkpoint_path}[/]")
