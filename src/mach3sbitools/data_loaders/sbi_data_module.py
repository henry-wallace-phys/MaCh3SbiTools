"""
PyTorch Lightning data module for SBI simulation datasets.

Wraps a pre-loaded :class:`~torch.utils.data.TensorDataset` in a
:class:`~lightning.LightningDataModule` so that Lightning's
:class:`~lightning.pytorch.trainer.Trainer` can manage train/validation
splitting and DataLoader construction consistently across all DDP ranks.

The dataset is expected to have been pre-loaded into CPU RAM via
:meth:`~mach3sbitools.data_loaders.ParaketDataset.to_tensor_dataset`
before this module is constructed. ``num_workers=0`` is used throughout
because the data is already a contiguous tensor in RAM and worker processes
would only add IPC overhead.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from mach3sbitools.utils.config import TrainingConfig


class SBIDataModule(L.LightningDataModule):
    """
    Lightning data module over a pre-loaded ``(theta, x)`` dataset.

    Performs a reproducible train/validation split in :meth:`setup` (called
    once per rank by the Trainer) and exposes standard DataLoaders.

    .. note::

        The random split uses a fixed seed of ``42`` so that all DDP ranks
        produce identical train and validation sets. If the seed is changed
        it must be changed consistently across all ranks, otherwise each rank
        will train and validate on different data.
    """

    def __init__(
        self,
        dataset: TensorDataset,
        config: TrainingConfig,
    ):
        """
        :param dataset: Pre-loaded ``(theta, x)``
            :class:`~torch.utils.data.TensorDataset` in CPU RAM, typically
            produced by
            :meth:`~mach3sbitools.data_loaders.ParaketDataset.to_tensor_dataset`.
        :param config: Training configuration supplying
            ``validation_fraction`` and ``batch_size``. See
            :class:`~mach3sbitools.utils.config.TrainingConfig`.
        """
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Split the dataset into train and validation subsets.

        Called by Lightning on every rank before the first epoch. The fixed
        generator seed ensures all ranks produce the same split regardless
        of the order in which processes reach this call.

        :param stage: Lightning stage string (``"fit"``, ``"validate"``,
            etc.). Accepted but unused — the split is always performed.
        """
        n_val = int(len(self.dataset) * self.config.validation_fraction)
        n_train = len(self.dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training DataLoader.

        Shuffles each epoch and drops the last incomplete batch to keep
        batch sizes uniform across DDP ranks, which is required when using
        :class:`~torch.nn.parallel.DistributedDataParallel`.

        :returns: Shuffling :class:`~torch.utils.data.DataLoader` over the
            training subset with ``pin_memory=True`` for fast GPU transfers.
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training set has not been set.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation DataLoader.

        :returns: Non-shuffling :class:`~torch.utils.data.DataLoader` over
            the validation subset with ``pin_memory=True``.
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation set has not been set.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
