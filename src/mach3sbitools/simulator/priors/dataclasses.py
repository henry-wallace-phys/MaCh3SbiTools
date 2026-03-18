from dataclasses import dataclass

import numpy as np
import torch


@dataclass(eq=False, repr=False)
class PriorData(torch.nn.Module):
    parameter_names: np.ndarray
    nominals: torch.Tensor
    covariance_matrix: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    def __post_init__(self):
        super().__init__()

    def __getitem__(self, mask: torch.Tensor) -> "PriorData":
        # Lets us do some masking

        # Need this for the parameter names
        np_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        return PriorData(
            parameter_names=self.parameter_names[np_mask],
            nominals=self.nominals[mask],
            covariance_matrix=self.covariance_matrix[mask][:, mask],
            lower_bounds=self.lower_bounds[mask],
            upper_bounds=self.upper_bounds[mask],
        )
