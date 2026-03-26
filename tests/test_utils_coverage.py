"""
Coverage-boosting tests for:
  - utils/device_handler.py  (lines 33, 48, 53-54)
  - utils/file_utils.py      (lines 37, 40, 68, 71, 97, 100)
  - utils/logger.py          (lines 106-112, 116, 120, 124, 128, 132, 140-142, 147)
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from mach3sbitools.utils.device_handler import TensorConversionError, TorchDeviceHandler
from mach3sbitools.utils.file_utils import (
    filter_nuisance,
    from_feather,
    to_feather,
)
from mach3sbitools.utils.logger import MaCh3Logger, get_logger

# ── TorchDeviceHandler ────────────────────────────────────────────────────────


class TestTorchDeviceHandler:
    def test_device_is_cpu_or_cuda(self):
        dh = TorchDeviceHandler()
        assert dh.device in ("cpu", "cuda")

    def test_to_tensor_from_numpy(self):
        dh = TorchDeviceHandler()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = dh.to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)

    def test_to_tensor_from_dataframe(self):
        """Covers the pd.DataFrame branch (line 48)."""
        dh = TorchDeviceHandler()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = dh.to_tensor(df)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2, 2)

    def test_to_tensor_from_list(self):
        """Covers the generic torch.tensor() branch (line 53)."""
        dh = TorchDeviceHandler()
        t = dh.to_tensor([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)

    def test_to_tensor_raises_on_bad_input(self):
        """Covers the TensorConversionError branch (line 54)."""
        dh = TorchDeviceHandler()
        with pytest.raises(TensorConversionError):
            dh.to_tensor(object())  # plain object() is not convertible

    @patch(
        "mach3sbitools.utils.device_handler.torch.cuda.is_available", return_value=True
    )
    def test_find_device_returns_cuda_when_available(self, mock_cuda):
        """Covers the cuda branch (line 33)."""
        dh = TorchDeviceHandler()
        assert dh.device == "cuda"


# ── file_utils ────────────────────────────────────────────────────────────────


class TestFilterNuisance:
    def test_raises_if_name_length_mismatch(self):
        """Covers line 37."""
        theta = np.ones((5, 3))
        with pytest.raises(ValueError):
            filter_nuisance(["a", "b"], ["a"], theta)  # 2 names, 3 cols

    def test_returns_theta_unchanged_when_nuisance_is_none(self):
        """Covers line 40 — the None early-return."""
        theta = np.ones((5, 3))
        result = filter_nuisance(["a", "b", "c"], None, theta)
        np.testing.assert_array_equal(result, theta)

    def test_filters_matching_params(self):
        theta = np.arange(10).reshape(2, 5).astype(np.float32)
        names = ["keep_1", "drop_x", "keep_2", "drop_y", "keep_3"]
        result = filter_nuisance(names, ["drop_*"], theta)
        assert result.shape == (2, 3)


class TestFromFeather:
    def test_raises_if_file_not_found(self):
        """Covers line 68."""
        with pytest.raises(FileNotFoundError):
            from_feather(Path("/no/such/file.feather"), ["a"])

    def test_loads_feather_without_nuisance(self, tmp_path):
        theta = np.ones((10, 3), dtype=np.float32)
        x = np.ones((10, 5), dtype=np.float32)
        path = tmp_path / "test.feather"
        to_feather(path, theta, x)

        t, xout = from_feather(path, ["a", "b", "c"])
        assert t.shape == (10, 3)
        assert xout.shape == (10, 5)

    def test_loads_feather_with_nuisance(self, tmp_path):
        """Covers line 71 — nuisance filtering inside from_feather."""
        theta = np.ones((10, 3), dtype=np.float32)
        x = np.ones((10, 5), dtype=np.float32)
        path = tmp_path / "test2.feather"
        to_feather(path, theta, x)

        t, _ = from_feather(path, ["keep", "drop_x", "keep2"], nuisance_pars=["drop_*"])
        assert t.shape == (10, 2)

    def test_accepts_string_path(self, tmp_path):
        """Covers the str->Path conversion branch."""
        theta = np.ones((4, 2), dtype=np.float32)
        x = np.ones((4, 3), dtype=np.float32)
        path = tmp_path / "str_path.feather"
        to_feather(path, theta, x)
        t, _ = from_feather(str(path), ["a", "b"])
        assert t.shape == (4, 2)


class TestToFeather:
    def test_raises_on_non_feather_suffix(self, tmp_path):
        """Covers line 97."""
        theta = np.ones((5, 2), dtype=np.float32)
        x = np.ones((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="feather"):
            to_feather(tmp_path / "out.csv", theta, x)

    def test_accepts_string_path(self, tmp_path):
        """Covers the str->Path conversion branch (line 100)."""
        theta = np.ones((4, 2), dtype=np.float32)
        x = np.ones((4, 3), dtype=np.float32)
        path = str(tmp_path / "out.feather")
        to_feather(path, theta, x)
        assert (tmp_path / "out.feather").exists()

    def test_writes_and_reads_round_trip(self, tmp_path):
        theta = np.random.rand(20, 4).astype(np.float32)
        x = np.random.rand(20, 6).astype(np.float32)
        path = tmp_path / "roundtrip.feather"
        to_feather(path, theta, x)
        t_out, x_out = from_feather(path, [f"p{i}" for i in range(4)])
        np.testing.assert_allclose(t_out, theta, rtol=1e-5)
        np.testing.assert_allclose(x_out, x, rtol=1e-5)


# ── MaCh3Logger ───────────────────────────────────────────────────────────────


class TestMaCh3Logger:
    def test_creates_logger(self):
        logger = MaCh3Logger(name="test_logger_basic")
        assert logger.logger.name == "test_logger_basic"

    def test_debug_method(self):
        """Covers line 116."""
        logger = MaCh3Logger(name="test_debug")
        with patch.object(logger._logger, "debug") as mock:
            logger.debug("hello %s", "world")
            mock.assert_called_once_with("hello %s", "world")

    def test_info_method(self):
        """Covers line 120."""
        logger = MaCh3Logger(name="test_info")
        with patch.object(logger._logger, "info") as mock:
            logger.info("info msg")
            mock.assert_called_once_with("info msg")

    def test_warning_method(self):
        """Covers line 124."""
        logger = MaCh3Logger(name="test_warning")
        with patch.object(logger._logger, "warning") as mock:
            logger.warning("warn msg")
            mock.assert_called_once_with("warn msg")

    def test_error_method(self):
        """Covers line 128."""
        logger = MaCh3Logger(name="test_error")
        with patch.object(logger._logger, "error") as mock:
            logger.error("err msg")
            mock.assert_called_once_with("err msg")

    def test_critical_method(self):
        """Covers line 132."""
        logger = MaCh3Logger(name="test_critical")
        with patch.object(logger._logger, "critical") as mock:
            logger.critical("crit msg")
            mock.assert_called_once_with("crit msg")

    def test_set_level(self):
        """Covers lines 140-142."""
        logger = MaCh3Logger(name="test_set_level")
        logger.set_level("DEBUG")  # should not raise

    def test_logger_property(self):
        """Covers line 147."""
        import logging

        logger = MaCh3Logger(name="test_prop")
        assert isinstance(logger.logger, logging.Logger)

    def test_writes_to_file(self, tmp_path):
        """Covers lines 106-112 — file handler branch."""
        log_file = tmp_path / "test.log"
        logger = MaCh3Logger(name="test_file_logger", log_file=log_file)
        logger.info("file log test")
        assert log_file.exists()

    def test_file_level_override(self, tmp_path):
        log_file = tmp_path / "debug.log"
        logger = MaCh3Logger(
            name="test_file_level", log_file=log_file, file_level="WARNING"
        )
        logger.warning("check")
        assert log_file.exists()

    def test_get_logger_returns_logger(self):
        import logging

        lg = get_logger("some_module")
        assert isinstance(lg, logging.Logger)
