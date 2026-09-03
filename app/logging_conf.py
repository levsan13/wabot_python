"""Simple readable logging, no extra dependencies."""

from __future__ import annotations

import logging
import sys


class _ColorFormatter(logging.Formatter):
    """Adds ANSI colors when the output is attached to a real terminal."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            if color:
                text = f"{color}{text}{self.RESET}"
        return text


def setup_logging(level: str = "INFO") -> None:
    """Install the root handler. Called once from the app lifespan."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter(use_color=sys.stdout.isatty()))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # libraries that are far too chatty for day-to-day use
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
