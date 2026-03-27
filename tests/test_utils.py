"""
Tests for mach3sbitools.utils.

Covers device_handler, file_utils, and logger.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from mach3sbitools.utils.device_handler import TensorConversionError, TorchDeviceHandler
from mach3sbitools.utils.file_utils import filter_nuisance, from_feather, to_feather
from mach3sbitools.utils.logger import MaCh3Logger, get_logger

# ─────────────────────────────────────────────────────────────────────────────
# TorchDeviceHandler
# ─────────────────────────────────────────────────────────────────────────────


class TestTorchDeviceHandler:
    def test_device_is_cpu_or_cuda(self):
        assert TorchDeviceHandler().device in ("cpu", "cuda")

    @patch(
        "mach3sbitools.utils.device_handler.torch.cuda.is_available", return_value=True
    )
    def test_finds_cuda_when_available(self, _):
        assert TorchDeviceHandler().device == "cuda"

    @pytest.mark.parametrize(
        "data,expected_shape",
        [
            (np.array([1.0, 2.0, 3.0], dtype=np.float32), (3,)),
            (pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), (2, 2)),
            ([1.0, 2.0, 3.0], (3,)),
        ],
    )
    def test_to_tensor_from_various_types(self, data, expected_shape):
        t = TorchDeviceHandler().to_tensor(data)
        assert isinstance(t, torch.Tensor)
        assert t.shape == torch.Size(expected_shape)

    def test_to_tensor_raises_on_unconvertible_object(self):
        with pytest.raises(TensorConversionError):
            TorchDeviceHandler().to_tensor(object())


# ─────────────────────────────────────────────────────────────────────────────
# filter_nuisance
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterNuisance:
    def test_raises_if_name_length_mismatch(self):
        with pytest.raises(ValueError):
            filter_nuisance(["a", "b"], ["a"], np.ones((5, 3)))

    def test_returns_theta_unchanged_when_nuisance_is_none(self):
        theta = np.ones((5, 3))
        np.testing.assert_array_equal(
            filter_nuisance(["a", "b", "c"], None, theta), theta
        )

    def test_filters_matching_params(self):
        theta = np.arange(10).reshape(2, 5).astype(np.float32)
        names = ["keep_1", "drop_x", "keep_2", "drop_y", "keep_3"]
        assert filter_nuisance(names, ["drop_*"], theta).shape == (2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Feather I/O — round-trip covers read + write; error paths tested separately
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatherIO:
    @pytest.fixture()
    def feather_file(self, tmp_path):
        """Write a small feather file and return its path."""
        theta = np.random.rand(20, 4).astype(np.float32)
        x = np.random.rand(20, 6).astype(np.float32)
        path = tmp_path / "data.feather"
        to_feather(path, theta, x)
        return path, theta, x

    def test_round_trip_preserves_values(self, feather_file):
        path, theta, x = feather_file
        t_out, x_out = from_feather(path, [f"p{i}" for i in range(4)])
        np.testing.assert_allclose(t_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)

    def test_round_trip_with_nuisance_filter(self, tmp_path):
        theta = np.ones((10, 3), dtype=np.float32)
        x = np.ones((10, 5), dtype=np.float32)
        path = tmp_path / "nuisance.feather"
        to_feather(path, theta, x)
        t, _ = from_feather(path, ["keep", "drop_x", "keep2"], nuisance_pars=["drop_*"])
        assert t.shape == (10, 2)

    def test_to_feather_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "str.feather")
        to_feather(
            path, np.ones((4, 2), dtype=np.float32), np.ones((4, 3), dtype=np.float32)
        )
        assert (tmp_path / "str.feather").exists()

    def test_from_feather_accepts_string_path(self, feather_file):
        path, _, _ = feather_file
        t, _ = from_feather(str(path), [f"p{i}" for i in range(4)])
        assert t.shape[1] == 4

    def test_to_feather_raises_on_non_feather_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="feather"):
            to_feather(
                tmp_path / "out.csv",
                np.ones((5, 2), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),
            )

    def test_from_feather_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            from_feather(Path("/no/such/file.feather"), ["a"])


# ─────────────────────────────────────────────────────────────────────────────
# MaCh3Logger
# ─────────────────────────────────────────────────────────────────────────────


class TestMaCh3Logger:
    def test_creates_logger_with_correct_name(self):
        assert MaCh3Logger(name="test_basic").logger.name == "test_basic"

    def test_logger_property_returns_logging_logger(self):
        import logging

        assert isinstance(MaCh3Logger(name="test_prop").logger, logging.Logger)

    def test_writes_to_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        MaCh3Logger(name="test_file", log_file=log_file).info("file log test")
        assert log_file.exists()

    def test_file_level_override(self, tmp_path):
        log_file = tmp_path / "debug.log"
        MaCh3Logger(name="test_file_level", log_file=log_file, file_level="WARNING")
        assert log_file.exists()

    def test_set_level_does_not_raise(self):
        MaCh3Logger(name="test_set_level").set_level("DEBUG")

    def test_get_logger_returns_logging_logger(self):
        import logging

        assert isinstance(get_logger("some_module"), logging.Logger)

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
    def test_log_methods_delegate_to_underlying_logger(self, level):
        logger = MaCh3Logger(name=f"test_{level}")
        with patch.object(logger._logger, level) as mock:
            getattr(logger, level)("msg")
            mock.assert_called_once_with("msg")
