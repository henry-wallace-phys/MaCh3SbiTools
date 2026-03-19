"""
Rich-backed logging utilities for mach3sbitools.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "repr.number": "bold cyan",
        "repr.str": "green",
    }
)

#: Shared :class:`~rich.console.Console` instance used by both the logger
#: and the progress bar so Rich can route log lines above the live display.
console = Console(theme=_THEME)


class MaCh3Logger:
    """
    Rich-backed logger for mach3sbitools.

    Wraps :class:`logging.Logger` with a :class:`~rich.logging.RichHandler`
    for coloured console output and an optional plain-text file handler.

    Usage in application code::

        logger = MaCh3Logger("mach3sbi", log_file="run.log", level="INFO")

    Usage in submodules::

        from mach3sbitools.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Loaded [bold]50k[/] pairs")
    """

    LEVELS: ClassVar[dict] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        name: str = "mach3sbi",
        level: str = "INFO",
        log_file: Path | None = None,
        file_level: str | None = None,
        show_path: bool = False,
    ):
        """
        :param name: Logger name shown in every log line.
        :param level: Console log level (``DEBUG`` / ``INFO`` / ``WARNING`` /
            ``ERROR`` / ``CRITICAL``).
        :param log_file: Optional path to write plain-text logs alongside the
            Rich console output.
        :param file_level: Log level for the file handler. Defaults to ``DEBUG``.
        :param show_path: Show source file and line number in console output.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        rich_handler = RichHandler(
            console=console,
            level=self.LEVELS.get(level.upper(), logging.INFO),
            show_time=True,
            show_level=True,
            show_path=show_path,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format=self.DATEFMT,
        )
        self._logger.addHandler(rich_handler)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(self.LEVELS.get((file_level or "DEBUG").upper(), logging.DEBUG))
            fh.setFormatter(logging.Formatter(self.FILE_FORMAT, datefmt=self.DATEFMT))
            self._logger.addHandler(fh)
            self._logger.info(f"Logging to file: [cyan]{log_file}[/]")

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log *msg* at DEBUG level."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log *msg* at INFO level."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log *msg* at WARNING level."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log *msg* at ERROR level."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log *msg* at CRITICAL level."""
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str) -> None:
        """
        Adjust the console handler log level at runtime.

        :param level: New log level string.
        """
        for handler in self._logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(self.LEVELS.get(level.upper(), logging.INFO))

    @property
    def logger(self) -> logging.Logger:
        """The underlying :class:`logging.Logger` for stdlib compatibility."""
        return self._logger


def get_logger(name: str = "mach3sbi") -> logging.Logger:
    """
    Return a named child logger for use in submodules.

    :param name: Logger name, typically ``__name__``.
    :returns: A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


@dataclass
class TrainingProgress:
    """
    Container returned by :func:`create_progress`.

    Wraps the single shared :class:`~rich.progress.Progress` instance and the
    two task IDs so the training loop can advance each bar independently.

    Using one ``Progress`` for both tasks means there is a single ``Live``
    context owning the terminal — log messages written through the shared
    :data:`console` are always printed *above* both bars without tearing.

    :param progress: The Rich ``Progress`` object.  Use as a context manager
        (``with training_progress.progress:``) for the duration of training.
    :param fit_task: Task ID for the overall fit bar.  Advance once per epoch.
    :param epoch_task: Task ID for the per-epoch batch bar.  Reset to zero at
        the start of each epoch, advance once per batch step.
    """

    progress: Progress
    fit_task: TaskID
    epoch_task: TaskID

    # ── convenience helpers used by the training loop ─────────────────────────

    def start_epoch(self, epoch: int, total_epochs: int, n_steps: int) -> None:
        """
        Reset the epoch bar and update the fit-bar description for *epoch*.

        Call once at the top of each epoch before iterating over batches.

        :param epoch: Current epoch number (1-based).
        :param total_epochs: Total number of epochs (for the description label).
        :param n_steps: Number of batch steps in this epoch.
        """
        self.progress.reset(self.epoch_task, total=n_steps)
        self.progress.update(
            self.fit_task,
            description=f"[green]Fit  epoch {epoch}/{total_epochs}",
        )

    def step_batch(self) -> None:
        """Advance the epoch (batch) bar by one step."""
        self.progress.advance(self.epoch_task)

    def finish_epoch(self, train_loss: float, val_loss: float) -> None:
        """
        Advance the fit bar by one epoch and annotate both bars with losses.

        :param train_loss: Mean training loss for the completed epoch.
        :param val_loss: Mean validation loss for the completed epoch.
        """
        self.progress.advance(self.fit_task)
        loss_str = f"train {train_loss:.4f}  val {val_loss:.4f}"
        self.progress.update(self.fit_task, extra=loss_str)
        self.progress.update(self.epoch_task, extra=loss_str)


def create_progress(
    *,
    total_epochs: int,
    steps_per_epoch: int,
    show_progress: bool = True,
    console: Console = console,
) -> TrainingProgress | nullcontext:
    """
    Build a two-level nested progress display for training.

    Returns a :class:`TrainingProgress` that renders two bar rows inside a
    single ``Live`` context::

        Fit   epoch 12/500  ━━━━━━━━╸━━━━━━━━━━━━━━━━  12/500 • 0:00:14 < 0:09:43 • train 1.2345  val 1.3456
        Epoch              ━━━━━━━━━━━━━━━━━━━━━━━━━╸━  97/100 • 0:00:01 < 0:00:00 • train 1.2345  val 1.3456

    Because both tasks share one ``Progress`` / ``Live`` instance, log
    messages written via :data:`console` are always routed *above* the bars.
    ``redirect_stdout`` / ``redirect_stderr`` catch any output that bypasses
    Rich (e.g. from PyTorch internals) and route it the same way.

    When *show_progress* is ``False`` a :func:`~contextlib.nullcontext` is
    returned so callers can use ``with create_progress(...):`` unconditionally.

    Typical usage in a training loop::

        tp = create_progress(total_epochs=500, steps_per_epoch=len(loader))
        with tp.progress:
            for epoch in range(1, 501):
                tp.start_epoch(epoch, 500, len(loader))
                for theta, x in loader:
                    ...
                    tp.step_batch()
                tp.finish_epoch(train_loss, val_loss)

    :param total_epochs: Maximum number of training epochs.
    :param steps_per_epoch: Number of batches per epoch.
    :param show_progress: ``False`` disables all display (CI / non-interactive).
    :param console: Rich console.  Must be the same instance used by
        :class:`MaCh3Logger` so log output is routed correctly.
    :returns: :class:`TrainingProgress` or :func:`~contextlib.nullcontext`.
    """
    if not show_progress:
        return nullcontext()

    _COLUMNS = [
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<24}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("• [dim]{task.fields[extra]}"),
    ]

    progress = Progress(
        *_COLUMNS,
        console=console,
        auto_refresh=True,
        refresh_per_second=10,
        # Route raw stdout/stderr through the Live context so anything that
        # bypasses Rich still appears above the bars rather than tearing them.
        redirect_stdout=True,
        redirect_stderr=True,
        transient=False,
    )

    fit_task = progress.add_task("[green]Fit", total=total_epochs, extra="")
    epoch_task = progress.add_task("[cyan]Epoch", total=steps_per_epoch, extra="")

    return TrainingProgress(progress=progress, fit_task=fit_task, epoch_task=epoch_task)
