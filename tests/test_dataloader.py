import pytest
import torch

from mach3sbitools.data_loaders import ParaketDataset
from mach3sbitools.utils import TorchDeviceHandler

device_handler = TorchDeviceHandler()


def test_generate_data(dummy_data_dir, test_consts):
    """Check if data has been generated correctly"""
    assert len([t for t in dummy_data_dir.glob("*feather")]) == test_consts.n_files


@pytest.fixture(scope="session")
def paraket_dataset(dummy_data_dir, test_consts):
    return ParaketDataset(dummy_data_dir, test_consts.parameter_names)


def test_data_loaded(paraket_dataset, test_consts):
    assert len(paraket_dataset) == test_consts.n_files


def test_get_item(paraket_dataset, test_consts):
    theta, x = paraket_dataset[0]
    test_x_tensor = device_handler.to_tensor(test_consts.x)
    test_theta_tensor = device_handler.to_tensor(test_consts.theta)
    torch.testing.assert_close(x, test_x_tensor)
    torch.testing.assert_close(theta, test_theta_tensor)


def test_nuisance_filter(dummy_data_dir, test_consts):
    nuisance = ["theta_1*"]
    filtered_set = ParaketDataset(dummy_data_dir, test_consts.parameter_names, nuisance)

    assert len(filtered_set) == test_consts.n_files

    theta_filt, _x_filt = filtered_set[0]
    assert len(theta_filt[0]) == test_consts.theta_dim - 11


def test_tensor_dataset(paraket_dataset, test_consts):
    """Check if data has been generated correctly"""
    data_set = paraket_dataset.to_tensor_dataset()
    assert len(data_set) == test_consts.n_files * test_consts.n_simulations
