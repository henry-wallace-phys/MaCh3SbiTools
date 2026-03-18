import logging
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeRemainingColumn, TaskID
from rich.theme import Theme

# Consistent theme across all console output
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

# Shared console instance — import this anywhere for consistent Rich output
console = Console(theme=_THEME)


class MaCh3Logger:
    """
    Rich-backed logger for mach3sbitools.

    Usage:
        logger = MaCh3Logger("mach3sbi", log_file="run.log", level="INFO")
        logger.info("Training started")
        logger.warning("Something looks off")
        logger.debug("Batch loss: 0.423")

    In submodules:
        from mach3sbitools.utils.logger import get_logger
        logger = get_logger(__name__)
    """

    LEVELS: ClassVar[dict] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Plain format for file output (no Rich markup)
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
        Args:
            name:       Logger name shown in every log line.
            level:      Console log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
            log_file:   Optional path to write plain-text logs.
            file_level: File log level; defaults to DEBUG (file gets everything).
            show_path:  Show the file/line source in console output (useful for debugging).
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)  # handlers do the filtering
        self._logger.propagate = False

        # ── Rich console handler ────────────────────────────────────────────
        rich_handler = RichHandler(
            console=console,
            level=self.LEVELS.get(level.upper(), logging.INFO),
            show_time=True,
            show_level=True,
            show_path=show_path,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,  # allows [bold red]text[/] in log messages
            log_time_format=self.DATEFMT,
        )
        self._logger.addHandler(rich_handler)

        # ── Plain file handler (optional) ───────────────────────────────────
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(self.LEVELS.get((file_level or "DEBUG").upper(), logging.DEBUG))
            fh.setFormatter(logging.Formatter(self.FILE_FORMAT, datefmt=self.DATEFMT))
            self._logger.addHandler(fh)
            self._logger.info(f"Logging to file: [cyan]{log_file}[/]")

    # ── Convenience pass-throughs ───────────────────────────────────────────

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str) -> None:
        """Adjust the console handler level at runtime."""
        for handler in self._logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(self.LEVELS.get(level.upper(), logging.INFO))

    @property
    def logger(self) -> logging.Logger:
        """Expose the underlying logger for stdlib logging compatibility."""
        return self._logger


def get_logger(name: str = "mach3sbi") -> logging.Logger:
    """
    Get a named child logger for use in submodules.
    Inherits handlers from whichever MaCh3Logger was initialised upstream.

    Usage (in any submodule):
        from mach3sbitools.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Loaded [bold]50k[/] pairs")
    """
    return logging.getLogger(name)


def create_progress(
    *,
    show_epoch: bool = True,
    total_epochs: int = 1,
    description: str = "Training",
    console: Console = console,
) -> tuple[Progress | nullcontext, TaskID | None]:
    """
    Returns a Progress object and optionally a pre-created epoch task.
    """
    if not show_epoch:
        return nullcontext(), None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        TextColumn("• Epoch {task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )

    epoch_task = progress.add_task(description, total=total_epochs)
    return progress, epoch_task
