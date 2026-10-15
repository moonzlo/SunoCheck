"""Log file for the windowed app, which has no console: journal lines and every uncaught exception.

Without the hooks an exception in a Qt slot or a worker thread of the exe vanishes (there is no stderr)."""

import logging
import sys
import threading
import traceback
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

LOG_NAME: str = "sunocheck.log"
LOG_MAX_BYTES: int = 1_000_000
LOG_BACKUPS: int = 3
LOG_FORMAT: str = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"

logger: logging.Logger = logging.getLogger("sunocheck")


def setup_log_file(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path: Path = log_dir / LOG_NAME
    handler = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path


def install_exception_hooks(report: Callable[[str], None]) -> None:
    """Uncaught exceptions of the main thread, Qt slots and threads are logged, then handed to `report`."""

    def on_exception(kind: type[BaseException], value: BaseException | None, tb: TracebackType | None) -> None:
        details: str = "".join(traceback.format_exception(kind, value, tb))
        logger.error("uncaught exception\n%s", details)
        report(details)

    def on_thread_exception(args: threading.ExceptHookArgs) -> None:
        on_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = on_exception
    threading.excepthook = on_thread_exception
