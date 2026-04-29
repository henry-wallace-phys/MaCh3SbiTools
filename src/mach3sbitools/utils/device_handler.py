"""
PyTorch device detection and tensor conversion utilities.
"""

import numpy as np
import pandas as pd
import torch


class TensorConversionError(Exception):
    """Raised when an object cannot be converted to a :class:`torch.Tensor`."""


class TorchDeviceHandler:
    """
    Detects the best available PyTorch device and provides tensor conversion.

    The device is detected once at construction time and cached.
    """

    _device: str

    def __init__(self):
        self._device = self._find_device()

    @property
    def device(self) -> str:
        """The detected device string, e.g. ``"cuda"`` or ``"cpu"``."""
        return self._device

    @staticmethod
    def _find_device() -> str:
        """Return ``"cuda"`` if available, otherwise ``"cpu"``."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def to_tensor(self, data) -> torch.Tensor:
        """
        Convert an array-like object to a :class:`torch.Tensor` on the active device.

        Handles :class:`~pandas.DataFrame`, :class:`~numpy.ndarray`, and any
        object accepted by :func:`torch.tensor`.

        :param data: Input data to convert.
        :returns: Float tensor on :attr:`device`.
        :raises TensorConversionError: If conversion fails.
        """
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values.astype(np.float32), device=self.device)
        if isinstance(data, np.ndarray):
            return torch.tensor(data.astype(np.float32), device=self.device)
        if isinstance(data, list):
            return torch.tensor(data, device=self.device)
        if isinstance(data, torch.Tensor):
            return data.clone().detach().to(self.device)
        try:
            return torch.tensor(data, device=self.device)
        except Exception as e:
            raise TensorConversionError(
                f"Cannot convert object of type {type(data)} to torch tensor"
            ) from e
