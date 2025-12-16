import torch
import numpy as np
import pandas as pd

class TensorConversionError(Exception):
    ...

class TorchDeviceHander:
    def __init__(self):
        self._device = self._find_device()
    
    @property
    def device(self):
        return self._device
    
    def _find_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon [let's be fancy!]
        else:
            return "cpu"

    def to_tensor(self, data)->torch.Tensor:
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values.astype(np.float32), device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data.astype(np.float32), device=self.device)
        try:
            return torch.tensor(data, device=self.device)
        except Exception as e:
            raise TensorConversionError(f"Cannot convert object of type {type(data)} to torch tensor") from e
