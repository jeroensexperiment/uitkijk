"""Keep camera passwords out of the log: rtsp://user:secret@host becomes rtsp://user:***@host.

Errors from FFmpeg and PyAV contain the full camera address, and logs get pasted into chats and
issues."""

from __future__ import annotations

import logging
import re

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CREDENTIALS = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+):[^\s'\"]*@")


def redact(text: str) -> str:
    return _CREDENTIALS.sub(r"\1:***@", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup(level: int | str = logging.INFO) -> None:
    """logging.basicConfig with passwords hidden in every line (tracebacks included)."""
    logging.basicConfig(level=level, format=FORMAT)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(RedactingFormatter(FORMAT))
