from unittest.mock import call, patch

import numpy as np
import pytest
import torch

from mach3sbitools.simulator.priors.dataclasses import PriorData
from mach3sbitools.simulator.priors.prior import _check_boundary
from mach3sbitools.utils import get_logger

logger = get_logger()


@pytest.mark.slow
def test_dist_types(prior):
    # Flat mask first
    assert len(prior._priors) == 3
    assert torch.sum(prior._priors[0].mask).item() == 1
    assert torch.sum(prior._priors[1].mask).item() == 3
    assert torch.sum(prior._priors[2].mask).item() == 26


def test_prior_data_slicing():
    param_names = np.array(["a", "b", "c"])
    nominals = torch.Tensor([1, 2, 3])
    covariance_matrix = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lower_bound = torch.Tensor([1, 2, 3])
    upper_bound = torch.Tensor([4, 5, 6])

    test_data = PriorData(
        param_names, nominals, covariance_matrix, lower_bound, upper_bound
    )

    mask = torch.tensor([True, False, True])
    np_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    sliced_data = test_data[mask]

    np.testing.assert_array_equal(sliced_data.parameter_names, param_names[np_mask])
    assert torch.equal(sliced_data.nominals, nominals[mask])
    assert torch.equal(sliced_data.covariance_matrix, covariance_matrix[mask][:, mask])
    assert torch.equal(sliced_data.lower_bounds, lower_bound[mask])
    assert torch.equal(sliced_data.upper_bounds, upper_bound[mask])


def test_check_boundary_warning():
    nominal = torch.tensor([1.0, 2.0])
    error = torch.tensor([0.1, 0.1])
    lower_bound = torch.tensor([-100.0, 2.0])
    upper_bound = torch.tensor([1.0, 100.0])
    parameter_names = np.array(["param_a", "param_b"])

    with patch.object(logger, "warning") as mock_warning:
        _check_boundary(nominal, error, lower_bound, upper_bound, parameter_names)

        expected_calls = [
            call(
                "The following parameters have boundaries > 10σ from their prior nominal"
            ),
            call(
                "   'param_a' | Nominal: 1.000000, Error 0.100000 | Lower Bnd -100.000000, Upper Bnd 1.000000"
            ),
            call(
                "   'param_b' | Nominal: 2.000000, Error 0.100000 | Lower Bnd 2.000000, Upper Bnd 100.000000"
            ),
        ]

        mock_warning.assert_has_calls(expected_calls, any_order=False)


def test_check_boundary_no_warning_when_within_bounds():
    nominal = torch.tensor([1.0, 2.0])
    error = torch.tensor([0.1, 0.1])
    # Bounds are within 10σ — lower_bound > nominal - 10*error, upper_bound < nominal + 10*error
    lower_bound = torch.tensor([0.5, 1.5])  # both > nominal - 1.0
    upper_bound = torch.tensor([1.5, 2.5])  # both < nominal + 1.0
    parameter_names = np.array(["param_a", "param_b"])

    with patch.object(logger, "warning") as mock_warning:
        _check_boundary(nominal, error, lower_bound, upper_bound, parameter_names)
        mock_warning.assert_not_called()
