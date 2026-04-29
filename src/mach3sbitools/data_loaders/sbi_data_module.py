"""
PyTorch Lightning data module for SBI simulation datasets.

Dataset sharing strategy
------------------------
The ``TensorDataset`` passed to this module lives in **CPU RAM** and is
**not copied per DDP rank**.  Lightning's built-in ``DistributedSampler``
(activated automatically when ``strategy="ddp"``) gives each rank a
disjoint slice of indices, so every GPU reads only its own share from the
shared tensor without any inter-process data replication.

This is the correct pattern for in-memory datasets under DDP:

* Each rank receives the full ``TensorDataset`` reference (shared memory).
* ``DistributedSampler`` partitions the index space; ``pin_memory=True``
  then pages only the required rows to GPU VRAM.
* There is zero redundant I/O or memory duplication.

``num_workers=0`` is kept throughout: spawning worker processes for an
already-RAM-resident tensor would only add IPC overhead.
"""

from __future__ import annotations

import warnings

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from mach3sbitools.utils.config import TrainingConfig

warnings.filterwarnings(
    "ignore",
    message=".*num_workers.*bottleneck.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*LeafSpec.*deprecated.*",
    category=UserWarning,
)


class SBIDataModule(L.LightningDataModule):
    """
    Lightning data module over a pre-loaded ``(theta, x)`` dataset.

    The dataset is expected to have been pre-loaded into CPU RAM via
    :meth:`~mach3sbitools.data_loaders.ParaketDataset.to_tensor_dataset`
    before this module is constructed.

    Under DDP, Lightning automatically wraps each DataLoader's sampler in a
    ``DistributedSampler``, which partitions the index space across ranks.
    Because the underlying ``TensorDataset`` tensors are kept in CPU shared
    memory (no ``.to(device)`` call on the dataset itself), each rank reads
    only its own slice — no data is copied between processes.

    .. note::

        The random split uses a fixed seed of ``42`` so that all DDP ranks
        produce identical train / validation index sets.  If you change this
        seed, change it consistently across all ranks.
    """

    def __init__(self, dataset: TensorDataset, config: TrainingConfig) -> None:
        """
        :param dataset: Pre-loaded ``(theta, x)`` :class:`~torch.utils.data.TensorDataset`
            in CPU RAM, produced by
            :meth:`~mach3sbitools.data_loaders.ParaketDataset.to_tensor_dataset`.
        :param config: Training configuration supplying ``validation_fraction``
            and ``batch_size``.
        """
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        warnings.filterwarnings(
            "ignore",
            message=".*num_workers.*bottleneck.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*LeafSpec.*",
            category=UserWarning,
        )
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

        Shuffling is handled by Lightning's ``DistributedSampler`` under DDP,
        or by ``shuffle=True`` on single-device runs.  ``drop_last=True``
        keeps batch sizes uniform across ranks.

        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training set has not been set; call setup() first.")
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

        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation set has not been set; call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
