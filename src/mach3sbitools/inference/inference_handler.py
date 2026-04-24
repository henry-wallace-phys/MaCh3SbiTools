"""
High-level inference interface.

Combines data loading, NPE model construction, training, and posterior
sampling into a single object. The typical workflow is::

    handler = InferenceHandler(prior_path)
    handler.set_dataset(data_folder)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    handler.train_posterior(training_config, model_config=posterior_config)
    samples = handler.sample_posterior(10_000, x_observed)

When loading a checkpoint produced by the above, no ``PosteriorConfig`` is
needed — it is read directly from the checkpoint::

    handler = InferenceHandler(prior_path)
    handler.load_posterior(checkpoint_path)
    samples = handler.sample_posterior(10_000, x_observed)
"""

import os
from pathlib import Path
from typing import cast

import lightning
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sbi.inference import NPE, DirectPosterior
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets import posterior_nn
from torch.utils.data import TensorDataset

from mach3sbitools.data_loaders import SBIDataModule
from mach3sbitools.data_loaders.paraket_dataloader import ParaketDataset
from mach3sbitools.inference.lightning_module import SBILightningModule
from mach3sbitools.simulator import load_prior
from mach3sbitools.types import SimulatorData
from mach3sbitools.utils.config import PosteriorConfig, TrainingConfig
from mach3sbitools.utils.device_handler import TorchDeviceHandler
from mach3sbitools.utils.logger import get_logger

logger = get_logger()
torch.set_float32_matmul_precision("medium")

torch.serialization.add_safe_globals([TrainingConfig, PosteriorConfig])


def _select_accelerator_and_strategy():
    if torch.cuda.is_available():
        return "gpu", "ddp"
    if torch.backends.mps.is_available():
        return "mps", "auto"  # DDP not supported
    return "cpu", "auto"


def _infer_dims_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """
    Infer (theta_dim, x_dim) from a density-estimator state dict.

    The structured z-score layers are the most reliable source because they
    are always present regardless of flow architecture (NSF, MAF, …):

    * ``net._embedding_net.0._mean``  shape ``[x_dim]``
    * ``net._transform._transforms.0._shift``  shape ``[theta_dim]``

    Falls back to a broad key scan so that checkpoints from unusual
    architectures still work when the canonical keys are absent.

    :raises ValueError: If dimensions cannot be determined.
    """
    # ── x_dim ────────────────────────────────────────────────────────────────
    x_dim: int | None = None
    if "net._embedding_net.0._mean" in state_dict:
        shape = state_dict["net._embedding_net.0._mean"].shape
        if shape:  # guard against scalar [] buffers (un-fitted model)
            x_dim = shape[0]

    # ── theta_dim ─────────────────────────────────────────────────────────────
    theta_dim: int | None = None
    if "net._transform._transforms.0._shift" in state_dict:
        shape = state_dict["net._transform._transforms.0._shift"].shape
        if shape:
            theta_dim = shape[0]

    # ── fallback: scan all keys for weight matrices ───────────────────────────
    # Different flow families (MAF / NSF / …) use different submodule names.
    # Rather than enumerate them all we look at every 2-D weight tensor and
    # use heuristic names to decide which end is theta vs x.
    if theta_dim is None or x_dim is None:
        for key, val in state_dict.items():
            if val.ndim != 2:
                continue
            if theta_dim is None and any(
                s in key
                for s in (
                    "autoregressive_net.initial_layer.weight",
                    "conditioner.weight",  # NSF conditioner
                    "spline_net.0.weight",  # some NSF variants
                )
            ):
                theta_dim = val.shape[1]
            if x_dim is None and any(
                s in key
                for s in (
                    "_embedding_net",
                    "context_net",
                    "embedding_net",
                )
            ):
                x_dim = val.shape[1]
            if theta_dim is not None and x_dim is not None:
                break

    if theta_dim is None or x_dim is None:
        missing = []
        if theta_dim is None:
            missing.append("theta_dim")
        if x_dim is None:
            missing.append("x_dim")
        raise ValueError(
            f"Could not infer {' or '.join(missing)} from checkpoint state dict. "
            "The checkpoint may be corrupt or from an incompatible sbi version. "
            f"Available keys: {list(state_dict.keys())[:20]}"
        )

    return theta_dim, x_dim


def _build_and_prime_density_estimator(
    inference: NPE,
    theta_dim: int,
    x_dim: int,
) -> nn.Module:
    """
    Build and fully prime a density estimator so its z-score buffers are
    correctly shaped before ``load_state_dict`` is called.

    sbi's ``Standardize`` layers (both in the embedding net and in the flow
    transform) are initialised as shape-``[]`` scalar buffers.  They are
    re-shaped the first time ``_build_neural_net`` is called with real-shaped
    tensors *and* the network is run forward.  We therefore build the net with
    dummy tensors of the right shape and then run a forward pass through each
    normalisation layer explicitly.

    :param inference: A freshly created :class:`sbi.inference.NPE` object
        whose ``_build_neural_net`` has not yet been called.
    :param theta_dim: Number of parameters (theta columns).
    :param x_dim: Number of observables (x columns).
    :returns: The density estimator with correctly shaped buffers, in eval
        mode on CPU.
    """
    dummy_theta = torch.zeros(2, theta_dim)
    dummy_x = torch.zeros(2, x_dim)

    # _build_neural_net calls append_simulations internally, which sets the
    # buffer shapes from the shapes of dummy_theta and dummy_x.
    density_estimator = inference._build_neural_net(dummy_theta, dummy_x)
    density_estimator.eval()

    with torch.no_grad():
        net = getattr(density_estimator, "net", None)
        if net is None:
            return density_estimator

        # Prime the embedding-net z-score layer (shapes x_dim buffers).
        embedding_net = getattr(net, "_embedding_net", None)
        if embedding_net is not None:
            try:
                embedding_net(dummy_x)
            except Exception:
                # Some embedding nets are identity / don't accept plain input;
                # _build_neural_net will have already sized the buffers.
                pass

        # Prime the flow z-score layer (shapes theta_dim buffers).
        # We cannot call density_estimator.forward() because NFlowsFlow
        # requires (theta, x) and performs the full expensive transform, but
        # we only need the first standardisation layer to run.
        transform = getattr(net, "_transform", None)
        if transform is not None:
            transforms = getattr(transform, "_transforms", [])
            if transforms:
                first = transforms[0]
                # Standardize.forward(theta) is safe to call directly.
                if hasattr(first, "_shift") and first._shift.shape:
                    # Already sized — nothing to do.
                    pass
                elif callable(first):
                    try:
                        first(dummy_theta)
                    except Exception:
                        pass

    return density_estimator


class InferenceHandler:
    """
    High-level interface for NPE training and posterior sampling.

    Manages the full inference pipeline: loading simulations from disk,
    building and training an NPE density estimator via PyTorch Lightning,
    and drawing posterior samples conditioned on observed data.
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
            Passed directly to
            :meth:`~mach3sbitools.simulator.Prior.set_nuisance_filter`.
        """
        self.device_handler = TorchDeviceHandler()

        self.prior = load_prior(prior_path)
        self.parameter_names = self.prior.prior_data.parameter_names
        self.nuisance_pars = nuisance_pars

        if len(self.parameter_names[self.prior._nuisance_filter]) != len(
            self.prior.prior_data.parameter_names
        ):
            raise ValueError(
                "Prior must have same nuisance params as inference handler!"
            )

        self.dataset: ParaketDataset | None = None
        self.inference: NPE | None = None
        self.posterior = None
        self._density_estimator: nn.Module | None = None
        self._tensor_dataset: TensorDataset | None = None

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
        Pre-load all feather files into RAM as a flat
        :class:`~torch.utils.data.TensorDataset`.

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

        :param config: Architecture and hyperparameter settings. See
            :class:`~mach3sbitools.utils.PosteriorConfig`.
        """
        neural_net = posterior_nn(
            model=config.model,
            hidden_features=config.hidden_features,
            num_transforms=config.num_transforms,
            dropout_probability=config.dropout_probability,
            num_blocks=config.num_blocks,
            num_bins=config.num_bins,
            z_score_x="structured",
            z_score_theta="structured",
        )
        self.inference = NPE(
            prior=self.prior,
            density_estimator=neural_net,
            device=self.device_handler.device,
        )
        logger.info(
            f"NPE created | {config.model} | "
            f"hidden=[cyan]{config.hidden_features}[/] "
            f"transforms=[cyan]{config.num_transforms}[/] "
            f"blocks=[cyan]{config.num_blocks}[/] "
            f"bins=[cyan]{config.num_bins}[/]"
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train_posterior(
        self,
        config: TrainingConfig,
        model_config: PosteriorConfig | None = None,
    ) -> None:
        """
        Train the density estimator using PyTorch Lightning.

        Constructs an
        :class:`~mach3sbitools.inference.lightning_module.SBILightningModule`
        and
        :class:`~mach3sbitools.inference.lightning_datamodule.SBIDataModule`,
        then delegates the full training loop to a
        :class:`~lightning.pytorch.trainer.Trainer` configured for multi-GPU
        DDP.

        The number of nodes is read from the ``SLURM_NNODES`` environment
        variable when present, falling back to ``1`` for single-node runs.
        GPU detection is automatic — Lightning will use all visible GPUs or
        fall back to CPU if none are available.

        Checkpoints are written by a
        :class:`~lightning.pytorch.callbacks.ModelCheckpoint` callback and
        are self-contained (architecture + weights). Early stopping monitors
        ``ema_val_loss`` with a patience of ``config.stop_after_epochs``.

        After training completes, :attr:`_density_estimator` is set to the
        model at the best checkpoint and placed in eval mode so
        :meth:`sample_posterior` can be called immediately.

        :param config: Training loop settings. See
            :class:`~mach3sbitools.utils.config.TrainingConfig`.
        :param model_config: Architecture configuration embedded in every
            checkpoint. When ``None`` the checkpoint will not be
            self-contained and ``posterior_config`` must be supplied to
            :meth:`load_posterior`.
        :raises ValueError: If :meth:`load_training_data` has not been called.
        :raises ValueError: If :meth:`create_posterior` has not been called.
        :raises ValueError: If ``config.resume_checkpoint`` is set but
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

        lightning_module = SBILightningModule(density_estimator, config, model_config)
        data_module = SBIDataModule(self._tensor_dataset, config)

        # Model checkpoint needs some overriding to save properly
        model_checkpoint = ModelCheckpoint(
            dirpath=(config.save_path.parent if config.save_path else None),
            filename=f"{config.save_path.stem if config.save_path else ''}"
            + "{epoch}-{ema_val_loss:.4f}",
            monitor="ema_val_loss",
            save_top_k=3,
            every_n_epochs=config.autosave_every,
            save_last=True,
        )

        model_checkpoint.CHECKPOINT_NAME_LAST = (
            str(config.save_path.stem) if config.save_path else "last"
        )

        callbacks = [
            EarlyStopping(
                monitor="ema_val_loss",
                patience=config.stop_after_epochs,
                mode="min",
            ),
            model_checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ]

        tb_logger = (
            TensorBoardLogger(save_dir=str(config.tensorboard_dir))
            if config.tensorboard_dir
            else True
        )

        acc, strat = _select_accelerator_and_strategy()

        trainer = lightning.Trainer(
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            logger=tb_logger,
            precision="bf16-mixed" if config.use_amp else "32-true",
            gradient_clip_val=1.0,
            enable_progress_bar=config.show_progress,
            log_every_n_steps=50,
            strategy=strat,
            accelerator=acc,
            devices="auto",
            num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        )

        trainer.fit(
            lightning_module,
            datamodule=data_module,
            ckpt_path=(
                str(config.resume_checkpoint) if config.resume_checkpoint else None
            ),
        )

        self._density_estimator = lightning_module.model
        self._density_estimator.eval()
        if config.save_path is None:
            raise FileNotFoundError("No checkpoint provided.")

        trainer.save_checkpoint(config.save_path)

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

        pars = DirectPosteriorParameters(enable_transform=False)

        self.posterior = self.inference.build_posterior(
            self._density_estimator, posterior_parameters=pars
        )

    def sample_posterior(
        self,
        num_samples: int,
        x: list[float] | np.ndarray,
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
            np.array([x]), dtype=torch.float32, device=self.device_handler.device
        )
        return cast(
            torch.Tensor, self.posterior.sample((num_samples,), x=x_tensor, **kwargs)
        )

    def load_posterior(
        self,
        checkpoint_path: Path,
        posterior_config: PosteriorConfig | None = None,
    ) -> None:
        """
        Load a trained density estimator from a checkpoint file.

        The ``PosteriorConfig`` is read from the checkpoint's
        ``"model_config"`` key when present. If *posterior_config* is also
        supplied alongside a checkpoint that already contains a
        ``"model_config"``, a warning is emitted and the checkpoint config
        is used — it is the authoritative source of truth for the model
        architecture, preventing silent mismatches.

        Supports both autosave checkpoints (dicts with ``"model_state"`` and
        ``"model_config"`` keys) and plain state dicts (legacy format that
        requires ``posterior_config`` to be supplied explicitly).

        The model dimensions (``theta_dim``, ``x_dim``) are always inferred
        directly from the checkpoint's state dict, never from the config.
        This guarantees that the z-score buffers (``_shift`` / ``_scale`` /
        ``_mean`` / ``_std``) are sized correctly before ``load_state_dict``
        is called, avoiding shape-mismatch ``RuntimeError``s.

        :param checkpoint_path: Path to a ``.pt`` / ``.ckpt`` checkpoint
            file.
        :param posterior_config: Ignored when the checkpoint contains a
            ``"model_config"`` entry (i.e. any checkpoint produced by a
            current trainer). Required for legacy plain state-dict
            checkpoints that pre-date self-contained saves.
        :raises FileNotFoundError: If *checkpoint_path* does not exist.
        :raises ValueError: If no model config can be found in the checkpoint
            and none was supplied.
        :raises ValueError: If dimensions cannot be inferred from the state
            dict.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # ── Unpack state dict and config from the checkpoint ──────────────────
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
            logger.info(f"Loading autosave checkpoint from epoch {ckpt.get('epoch')}")

            ckpt_config: PosteriorConfig | None = ckpt.get("model_config")

            if ckpt_config is not None and posterior_config is not None:
                logger.warning(
                    "Both a checkpoint model_config and a posterior_config were "
                    "provided. The checkpoint config will be used — the supplied "
                    "config is ignored."
                )

            # Always prefer the config baked into the checkpoint.
            config: PosteriorConfig | None = ckpt_config or posterior_config

            if config is None:
                raise ValueError(
                    "Checkpoint does not contain a model_config and none was "
                    "supplied. Pass posterior_config= explicitly for old checkpoints."
                )
        else:
            # Legacy plain state dict.
            state_dict = ckpt
            config = posterior_config
            logger.info("Loading plain state-dict checkpoint")
            if config is None:
                raise ValueError(
                    "Plain state-dict checkpoint contains no model_config. "
                    "Pass posterior_config= explicitly."
                )

        # Strip torch.compile prefix if present.
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }
            logger.debug("Stripped _orig_mod. prefix from compiled model state dict")

        # ── Infer dimensions from the checkpoint itself ───────────────────────
        # We read theta_dim and x_dim from the state dict rather than from the
        # config so that the z-score buffers are always sized to match the
        # weights that are about to be loaded, regardless of what the config
        # says.
        theta_dim, x_dim = _infer_dims_from_state_dict(state_dict)
        logger.debug(f"Inferred from checkpoint: theta_dim={theta_dim}, x_dim={x_dim}")

        # ── Rebuild the network with correctly shaped buffers ─────────────────
        # create_posterior() constructs a fresh NPE object (self.inference).
        # _build_and_prime_density_estimator() then calls
        # inference._build_neural_net() with real-shaped dummy tensors so that
        # sbi's Standardize layers allocate 1-D buffers of the right length,
        # and then runs the normalisation layers forward once to ensure the
        # buffers are not still scalar [] tensors.
        self.create_posterior(config)

        if self.inference is None:
            raise ValueError("create_posterior() did not initialise self.inference.")

        density_estimator = _build_and_prime_density_estimator(
            self.inference, theta_dim, x_dim
        )

        # ── Load weights ──────────────────────────────────────────────────────
        density_estimator.load_state_dict(state_dict)
        density_estimator.to(self.device_handler.device).eval()
        self._density_estimator = density_estimator

        logger.info(f"Density estimator loaded from [cyan]{checkpoint_path}[/]")

    def get_log_likelihood(
        self, theta: SimulatorData, x: list[float] | np.ndarray, **kwargs
    ) -> torch.Tensor:
        """
        Evaluate the log-likelihood of *theta* given observed data *x*.

        :param theta: Sampled parameter array of shape
            ``(n_samples, n_params)``.
        :param x: Observed data vector *x_o*.
        :param kwargs: Additional keyword arguments forwarded to
            ``sbi.posterior.log_prob``.
        :raises ValueError: If no density estimator has been trained or loaded.
        :returns: Log-probability tensor of shape ``(n_samples,)``.
        """
        self.build_posterior()

        if self.posterior is None:
            raise ValueError("Train or load a density estimator first.")

        x_tensor = torch.tensor(
            np.array([x]), dtype=torch.float32, device=self.device_handler.device
        )
        theta_tensor = torch.tensor(
            np.array(theta), dtype=torch.float32, device=self.device_handler.device
        )

        posterior = cast(DirectPosterior, self.posterior)

        return cast(
            torch.Tensor, posterior.log_prob(theta=theta_tensor, x=x_tensor, **kwargs)
        )
