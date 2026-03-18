"""
Data containers for prior parameters.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(eq=False, repr=False)
class PriorData(torch.nn.Module):
    """
    Dataclass holding the raw prior parameter arrays.

    Subclasses :class:`torch.nn.Module` so that tensors can be moved to a
    device via standard PyTorch mechanisms.

    :param parameter_names: Array of parameter name strings, shape ``(n_params,)``.
    :param nominals: Nominal (mean) values, shape ``(n_params,)``.
    :param covariance_matrix: Full covariance matrix, shape ``(n_params, n_params)``.
    :param lower_bounds: Hard lower bounds, shape ``(n_params,)``.
    :param upper_bounds: Hard upper bounds, shape ``(n_params,)``.
    """

    parameter_names: np.ndarray
    nominals: torch.Tensor
    covariance_matrix: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    def __post_init__(self):
        super().__init__()

    def __getitem__(self, mask: torch.Tensor) -> "PriorData":
        """
        Return a masked subset of the prior data.

        :param mask: Boolean tensor of shape ``(n_params,)``.
        :returns: New :class:`PriorData` containing only the selected parameters.
        """
        np_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        return PriorData(
            parameter_names=self.parameter_names[np_mask],
            nominals=self.nominals[mask],
            covariance_matrix=self.covariance_matrix[mask][:, mask],
            lower_bounds=self.lower_bounds[mask],
            upper_bounds=self.upper_bounds[mask],
        )
