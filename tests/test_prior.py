import pytest

import torch
import numpy as np

from mach3sbitools.simulator import (Prior,
                                     create_prior,
                                     load_prior)
from mach3sbitools.simulator.simulator_injector import get_simulator
from mach3sbitools.simulator.priors.dataclasses import PriorData

@pytest.fixture(scope="session")
def simulator_injector(simulator_module, simulator_class, dummy_config):
    return get_simulator(simulator_module, simulator_class, dummy_config)


# First let's check a flat/non-flat prior
@pytest.fixture
def prior(simulator_injector):
    # Need the injector
    return create_prior(simulator_injector)

def test_dist_types(prior):
    # Flat mask first
    assert len(prior._priors) == 2
    assert torch.sum(prior._priors[0].mask).item() == 3
    assert torch.sum(prior._priors[1].mask).item() == 7

def test_prior_data_slicing():
    param_names = np.array(['a', 'b', 'c'])
    nominals    = torch.Tensor([1, 2, 3])
    covariance_matrix = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lower_bound = torch.Tensor([1, 2, 3])
    upper_bound = torch.Tensor([4, 5, 6])

    test_data = PriorData(param_names,
                          nominals,
                          covariance_matrix,
                          lower_bound,
                          upper_bound)

    mask = torch.tensor([True, False, True])
    np_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    sliced_data = test_data[mask]

    np.testing.assert_array_equal(sliced_data.parameter_names, param_names[np_mask])
    assert torch.equal(sliced_data.nominals, nominals[mask])
    assert torch.equal(sliced_data.covariance_matrix, covariance_matrix[mask][:, mask])
    assert torch.equal(sliced_data.lower_bounds, lower_bound[mask])
    assert torch.equal(sliced_data.upper_bounds, upper_bound[mask])