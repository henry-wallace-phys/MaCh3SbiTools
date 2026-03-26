"""
Coverage-boosting tests for inference/tensorboard_writer.py (currently 19%).

Mocks SummaryWriter so no real TensorBoard is needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from mach3sbitools.inference.tensorboard_writer import TensorBoardWriter


@pytest.fixture()
def mock_writer():
    """TensorBoardWriter with SummaryWriter mocked out."""
    with patch(
        "mach3sbitools.inference.tensorboard_writer.SummaryWriter"
    ) as mock_sw_cls:
        mock_sw = MagicMock()
        mock_sw_cls.return_value = mock_sw
        writer = TensorBoardWriter(log_dir="/tmp/tb_test", device_type="cpu")
        yield writer, mock_sw


class TestTensorBoardWriterInit:
    def test_creates_summary_writer(self):
        with patch(
            "mach3sbitools.inference.tensorboard_writer.SummaryWriter"
        ) as mock_sw_cls:
            TensorBoardWriter(log_dir="/tmp/test", device_type="cpu")
            mock_sw_cls.assert_called_once_with(log_dir="/tmp/test")

    def test_stores_device_type(self):
        with patch("mach3sbitools.inference.tensorboard_writer.SummaryWriter"):
            writer = TensorBoardWriter(log_dir="/tmp/test", device_type="cuda")
            assert writer.device_type == "cuda"


class TestAddToWriter:
    def test_adds_loss_scalars(self, mock_writer):
        writer, sw = mock_writer
        opt = MagicMock()
        opt.param_groups = [{"lr": 1e-3}]

        writer.add_to_writer(
            epoch=1,
            train_loss=1.0,
            val_loss=1.1,
            ema_val_loss=1.05,
            best_val_loss=1.0,
            optimizer=opt,
            elapsed=2.5,
            epochs_no_improve=0,
            total_samples=100,
        )

        # Check key scalar calls were made
        calls = [c[0][0] for c in sw.add_scalar.call_args_list]
        assert "loss/train" in calls
        assert "loss/val" in calls
        assert "loss/val_ema" in calls
        assert "loss/best_val_ema" in calls
        assert "loss/train_val_gap" in calls

    def test_adds_lr_scalar(self, mock_writer):
        writer, sw = mock_writer
        opt = MagicMock()
        opt.param_groups = [{"lr": 5e-4}, {"lr": 1e-4}]

        writer.add_to_writer(
            epoch=2,
            train_loss=0.5,
            val_loss=0.6,
            ema_val_loss=0.55,
            best_val_loss=0.5,
            optimizer=opt,
            elapsed=1.0,
            epochs_no_improve=0,
            total_samples=50,
        )

        calls = [c[0][0] for c in sw.add_scalar.call_args_list]
        assert "lr/group_0" in calls
        assert "lr/group_1" in calls

    def test_adds_throughput_scalars(self, mock_writer):
        writer, sw = mock_writer
        opt = MagicMock()
        opt.param_groups = [{"lr": 1e-3}]

        writer.add_to_writer(
            epoch=1,
            train_loss=1.0,
            val_loss=1.1,
            ema_val_loss=1.0,
            best_val_loss=1.0,
            optimizer=opt,
            elapsed=4.0,
            epochs_no_improve=2,
            total_samples=200,
        )

        calls = [c[0][0] for c in sw.add_scalar.call_args_list]
        assert "throughput/samples_per_sec" in calls
        assert "throughput/epoch_seconds" in calls
        assert "early_stopping/epochs_no_improve" in calls

    def test_adds_gpu_scalars(self, mock_writer):
        writer, sw = mock_writer
        opt = MagicMock()
        opt.param_groups = [{"lr": 1e-3}]

        writer.add_to_writer(
            epoch=1,
            train_loss=1.0,
            val_loss=1.1,
            ema_val_loss=1.0,
            best_val_loss=1.0,
            optimizer=opt,
            elapsed=1.0,
            epochs_no_improve=0,
            total_samples=100,
        )

        calls = [c[0][0] for c in sw.add_scalar.call_args_list]
        assert "gpu/allocated_mb" in calls
        assert "gpu/reserved_mb" in calls


class TestGetGpuStats:
    def test_returns_zeros_on_cpu(self):
        with patch("mach3sbitools.inference.tensorboard_writer.SummaryWriter"):
            writer = TensorBoardWriter(log_dir="/tmp/t", device_type="cpu")
            stats = writer.get_gpu_stats()
            assert stats["allocated_mb"] == 0
            assert stats["reserved_mb"] == 0
            assert stats["max_reserved_mb"] == 0
            assert stats["memory_utilization_pct"] == 0

    def test_returns_zeros_when_cuda_unavailable(self):
        with patch("mach3sbitools.inference.tensorboard_writer.SummaryWriter"):
            with patch(
                "mach3sbitools.inference.tensorboard_writer.torch.cuda.is_available",
                return_value=False,
            ):
                writer = TensorBoardWriter(log_dir="/tmp/t", device_type="cuda")
                stats = writer.get_gpu_stats()
                assert stats["allocated_mb"] == 0

    def test_returns_memory_stats_on_cuda(self):
        with patch("mach3sbitools.inference.tensorboard_writer.SummaryWriter"):
            with patch(
                "mach3sbitools.inference.tensorboard_writer.torch.cuda.is_available",
                return_value=True,
            ):
                with patch(
                    "mach3sbitools.inference.tensorboard_writer.torch.cuda.memory_allocated",
                    return_value=1024**2 * 100,
                ):
                    with patch(
                        "mach3sbitools.inference.tensorboard_writer.torch.cuda.memory_reserved",
                        return_value=1024**2 * 200,
                    ):
                        with patch(
                            "mach3sbitools.inference.tensorboard_writer.torch.cuda.max_memory_reserved",
                            return_value=1024**2 * 200,
                        ):
                            writer = TensorBoardWriter(
                                log_dir="/tmp/t", device_type="cuda"
                            )
                            stats = writer.get_gpu_stats()
                            assert stats["allocated_mb"] == pytest.approx(100.0)
                            assert stats["reserved_mb"] == pytest.approx(200.0)
                            assert stats["memory_utilization_pct"] == pytest.approx(
                                50.0
                            )


class TestClose:
    def test_close_flushes_and_closes(self, mock_writer):
        writer, sw = mock_writer
        writer.close()
        sw.flush.assert_called_once()
        sw.close.assert_called_once()
