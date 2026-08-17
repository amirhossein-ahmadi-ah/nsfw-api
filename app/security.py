"""
Filesystem path security.

The API accepts a raw file path from the caller (the Telegram bot, running
on the same host). Because that path is attacker-influenceable in principle
(e.g. if the bot ever forwards a user-controlled string), we must ensure it
can never be used to read files outside a small allow-list of directories.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from app.config import settings


class PathSecurityError(Exception):
    """Raised when a requested path fails security validation."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _resolved_allowed_dirs() -> list:
    return settings.RESOLVED_ALLOWED_DIRS


def resolve_safe_path(raw_path: str) -> Path:
    """
    Resolve `raw_path` and verify it lives inside one of the configured
    allowed directories. Raises PathSecurityError on any violation.

    This blocks path traversal (../..), symlink escapes, and absolute paths
    pointing outside the allow-list.
    """
    if not raw_path or not isinstance(raw_path, str):
        raise PathSecurityError("INVALID_PATH", "Path must be a non-empty string.")

    try:
        candidate = Path(raw_path).resolve()
    except (OSError, RuntimeError) as exc:
        raise PathSecurityError("INVALID_PATH", f"Could not resolve path: {exc}")

    allowed_dirs = _resolved_allowed_dirs()
    if not allowed_dirs:
        raise PathSecurityError(
            "SERVER_MISCONFIGURED", "No allowed directories are configured."
        )

    for allowed in allowed_dirs:
        try:
            candidate.relative_to(allowed)
            return candidate
        except ValueError:
            continue

    raise PathSecurityError(
        "PATH_NOT_ALLOWED",
        "Requested path is outside the allowed directories.",
    )


def validate_file(path: Path) -> Tuple[str, int]:
    """
    Validate that the resolved path exists, is a regular file, has a
    supported extension, and is within the configured size limit.

    Returns (extension, size_bytes) on success.
    """
    if not path.exists():
        raise PathSecurityError("FILE_NOT_FOUND", "File does not exist.")

    if not path.is_file():
        raise PathSecurityError("NOT_A_FILE", "Requested path is not a regular file.")

    ext = path.suffix.lower()
    if ext not in settings.ALL_EXTENSIONS:
        raise PathSecurityError(
            "UNSUPPORTED_FORMAT", f"File extension '{ext}' is not supported."
        )

    size_bytes = os.path.getsize(path)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise PathSecurityError(
            "FILE_TOO_LARGE",
            f"File size {size_bytes} bytes exceeds limit of {max_bytes} bytes.",
        )

    if size_bytes == 0:
        raise PathSecurityError("EMPTY_FILE", "File is empty.")

    return ext, size_bytes
