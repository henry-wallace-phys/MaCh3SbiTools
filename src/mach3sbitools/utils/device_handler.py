import numpy as np
import pandas as pd
import torch


class TensorConversionError(Exception): ...


class TorchDeviceHandler:
    """
    Handles the device casting + tracking
    """
    def __init__(self):
        """
        Constructor
        """
        self._device = self._find_device()

    @property
    def device(self):
        """
        Get the current device
        :return:
        """
        return self._device

    @staticmethod
    def _find_device():
        """
        Just checks to see if we have a GPU + caches it
        :return:
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def to_tensor(self, data) -> torch.Tensor:
        """
        Casts some iterable to a torch tensor
        :param data:
        :return:
        """
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values.astype(np.float32), device=self.device)
        if isinstance(data, np.ndarray):
            return torch.tensor(data.astype(np.float32), device=self.device)
        try:
            return torch.tensor(data, device=self.device)
        except Exception as e:
            raise TensorConversionError(
                f"Cannot convert object of type {type(data)} to torch tensor"
            ) from e
