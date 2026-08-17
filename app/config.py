"""
Central configuration for the NSFW/NSFL classification API.

All tunable values are read from environment variables (typically supplied
via a .env file loaded with python-dotenv). Nothing here is hard-coded
downstream in the application -- always import `settings` from this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from the project root (one directory above /app).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


class Settings:
    """Immutable application settings, populated once at import time."""

    # --- Paths -------------------------------------------------------
    PROJECT_ROOT: Path = _PROJECT_ROOT
    MODEL_PATH: Path = Path(
        os.getenv("MODEL_PATH", str(_PROJECT_ROOT / "models" / "image-safety-classifier-l.onnx"))
    )
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(_PROJECT_ROOT / "logs")))

    # FFmpeg is a system dependency. Leave blank to use PATH; optionally
    # point these at custom executables on Windows/Linux.
    FFMPEG_BINARY: str = os.getenv("FFMPEG_BINARY", "").strip() or ""
    FFPROBE_BINARY: str = os.getenv("FFPROBE_BINARY", "").strip() or ""

    # Directories the API is allowed to read files from. Anything outside
    # these (after resolving symlinks / "..") is rejected with HTTP 403.
    ALLOWED_DIRS: List[str] = _get_list(
        "ALLOWED_DIRS", ["./files"]
    )

    # --- Server --------------------------------------------------------
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = _get_int("PORT", 8001)

    # --- Decision thresholds -------------------------------------------
    NSFW_THRESHOLD: float = _get_float("NSFW_THRESHOLD", 0.85)
    MIN_SUSPICIOUS_RATIO: float = _get_float("MIN_SUSPICIOUS_RATIO", 0.15)
    MIN_CONSECUTIVE_FRAMES: int = _get_int("MIN_CONSECUTIVE_FRAMES", 3)

    # Ratio-based flagging additionally requires at least this many
    # ABSOLUTE suspicious frames (not just a ratio). Without this, a tiny
    # sample (e.g. 1 suspicious frame out of 5 total = 20% ratio) would
    # trip MIN_SUSPICIOUS_RATIO on a single isolated frame -- exactly the
    # false-positive pattern this system exists to avoid. This has no
    # effect on the consecutive-run rule, which already has its own
    # absolute floor (MIN_CONSECUTIVE_FRAMES).
    MIN_SUSPICIOUS_ABSOLUTE_COUNT: int = _get_int("MIN_SUSPICIOUS_ABSOLUTE_COUNT", 3)

    # NSFL uses its own threshold/ratio -- default to mirroring NSFW unless
    # explicitly overridden, since NSFL content is generally treated with at
    # least equal caution.
    NSFL_THRESHOLD: float = _get_float("NSFL_THRESHOLD", NSFW_THRESHOLD)
    MIN_NSFL_SUSPICIOUS_RATIO: float = _get_float(
        "MIN_NSFL_SUSPICIOUS_RATIO", MIN_SUSPICIOUS_RATIO
    )
    MIN_NSFL_CONSECUTIVE_FRAMES: int = _get_int(
        "MIN_NSFL_CONSECUTIVE_FRAMES", MIN_CONSECUTIVE_FRAMES
    )
    MIN_NSFL_SUSPICIOUS_ABSOLUTE_COUNT: int = _get_int(
        "MIN_NSFL_SUSPICIOUS_ABSOLUTE_COUNT", MIN_SUSPICIOUS_ABSOLUTE_COUNT
    )

    # A single-frame override: if ANY frame exceeds this very high score,
    # flag regardless of ratio/consecutive rules. Set to >1.0 to disable.
    SINGLE_FRAME_OVERRIDE: float = _get_float("SINGLE_FRAME_OVERRIDE", 0.98)

    # --- Sampling / limits ----------------------------------------------
    MAX_MEDIA_FRAMES: int = _get_int("MAX_MEDIA_FRAMES", 48)
    MIN_MEDIA_FRAMES: int = _get_int("MIN_MEDIA_FRAMES", 8)
    SAMPLE_INTERVAL_SECONDS: float = _get_float("SAMPLE_INTERVAL_SECONDS", 10.0)
    FFPROBE_TIMEOUT: float = _get_float("FFPROBE_TIMEOUT", 10.0)
    FFMPEG_FRAME_TIMEOUT: float = _get_float("FFMPEG_FRAME_TIMEOUT", 8.0)
    MAX_FILE_SIZE_MB: int = _get_int("MAX_FILE_SIZE_MB", 100)

    # --- Concurrency / timeouts ------------------------------------------
    MAX_CONCURRENT_INFERENCE: int = _get_int("MAX_CONCURRENT_INFERENCE", 1)
    INFERENCE_BATCH_SIZE: int = _get_int("INFERENCE_BATCH_SIZE", 4)
    QUEUE_WAIT_TIMEOUT: float = _get_float("QUEUE_WAIT_TIMEOUT", 30.0)
    DETECTION_TIMEOUT: float = _get_float("DETECTION_TIMEOUT", 60.0)

    # --- Logging ----------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_MAX_BYTES: int = _get_int("LOG_MAX_BYTES", 10 * 1024 * 1024)  # 10 MB
    LOG_BACKUP_COUNT: int = _get_int("LOG_BACKUP_COUNT", 5)

    # --- Model info ---------------------------------------------------
    MODEL_REPO: str = os.getenv("MODEL_REPO", "OwenElliott/image-safety-classifier-l")
    MODEL_FILE: str = os.getenv("MODEL_FILE", "onnx/image-safety-classifier-l.onnx")
    MODEL_INPUT_SIZE: int = _get_int("MODEL_INPUT_SIZE", 224)

    # --- Supported formats -----------------------------------------------
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    GIF_EXTENSIONS = {".gif"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

    @property
    def RESOLVED_ALLOWED_DIRS(self) -> list[Path]:
        resolved = []
        for raw in self.ALLOWED_DIRS:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = self.PROJECT_ROOT / p
            try:
                resolved.append(p.resolve())
            except (OSError, RuntimeError):
                continue
        return resolved

    @property
    def ALL_EXTENSIONS(self) -> set:
        return self.IMAGE_EXTENSIONS | self.GIF_EXTENSIONS | self.VIDEO_EXTENSIONS


settings = Settings()
