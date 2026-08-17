"""Fast, bounded media sampling using FFmpeg/ffprobe.

The important performance property is that long videos are *not* decoded from
frame 0 to EOF. We probe duration and then perform a bounded number of input
seeks, decoding only one frame around each selected timestamp.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
import ffmpeg
from PIL import Image

from app.config import settings

logger = logging.getLogger("nsfw_api.media")


class MediaProbeError(ValueError):
    pass


def _which(name: str) -> str | None:
    return shutil.which(name)


def ffmpeg_binary() -> str:
    value = settings.FFMPEG_BINARY or _which("ffmpeg")
    if not value:
        raise MediaProbeError(
            "FFmpeg was not found. Install FFmpeg and ensure ffmpeg is on PATH."
        )
    return value


def ffprobe_binary() -> str:
    value = settings.FFPROBE_BINARY or _which("ffprobe")
    if not value:
        raise MediaProbeError(
            "ffprobe was not found. Install FFmpeg and ensure ffprobe is on PATH."
        )
    return value


def classify_media_type(ext: str) -> str:
    if ext in settings.IMAGE_EXTENSIONS:
        return "image"
    if ext in settings.GIF_EXTENSIONS:
        return "gif"
    if ext in settings.VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unrecognized extension: {ext}")


def _run_probe(path: Path) -> dict:
    cmd = [
        ffprobe_binary(),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        os.fspath(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=settings.FFPROBE_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise MediaProbeError("ffprobe executable could not be started.") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError("ffprobe timed out while inspecting the media.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise MediaProbeError(f"Could not probe media: {detail or 'invalid media'}") from exc

    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid metadata.") from exc


def _duration_seconds(metadata: dict) -> float:
    fmt = metadata.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        return duration

    for stream in metadata.get("streams") or []:
        try:
            duration = float(stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return 0.0


def probe_duration(path: Path) -> float:
    """Return duration without decoding media frames."""
    duration = _duration_seconds(_run_probe(path))
    if duration <= 0:
        raise MediaProbeError("Could not determine media duration.")
    return duration


def sample_count(duration: float) -> int:
    """Choose a bounded sample count from duration.

    Short clips get denser temporal coverage. Long clips are deliberately
    capped so inference cost remains predictable.
    """
    if duration <= 0:
        return 1
    target = math.ceil(duration / settings.SAMPLE_INTERVAL_SECONDS)
    return max(settings.MIN_MEDIA_FRAMES, min(settings.MAX_MEDIA_FRAMES, target))


def even_sample_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = max(1, count)
    if count == 1:
        return [min(0.05, duration / 2)]
    # Center samples inside equal temporal bins. This avoids repeatedly
    # hitting the exact first/last boundary where some codecs are awkward.
    step = duration / count
    return [min(duration - 0.001, (i + 0.5) * step) for i in range(count)]


def _decode_one_frame(path: Path, timestamp: float) -> np.ndarray | None:
    """Seek to timestamp and decode exactly one 224x224 RGB frame."""
    expected = settings.MODEL_INPUT_SIZE * settings.MODEL_INPUT_SIZE * 3
    try:
        stream = ffmpeg.input(os.fspath(path), ss=max(0.0, timestamp))
        stream = ffmpeg.output(
            stream,
            "pipe:1",
            vframes=1,
            vf=f"scale={settings.MODEL_INPUT_SIZE}:{settings.MODEL_INPUT_SIZE}:flags=bilinear",
            format="rawvideo",
            pix_fmt="rgb24",
        )
        process = ffmpeg.run_async(
            stream,
            cmd=ffmpeg_binary(),
            pipe_stdout=True,
            pipe_stderr=True,
            quiet=True,
            overwrite_output=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=settings.FFMPEG_FRAME_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    except (ffmpeg.Error, subprocess.TimeoutExpired, OSError) as exc:
        detail = ""
        if isinstance(exc, ffmpeg.Error) and getattr(exc, "stderr", None):
            detail = exc.stderr.decode("utf-8", "replace").strip()[-240:]
        logger.debug("FFmpeg frame seek failed at %.3fs: %s", timestamp, detail or exc)
        return None

    if len(stdout) < expected:
        return None
    return np.frombuffer(stdout[:expected], dtype=np.uint8).reshape(
        settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE, 3
    ).copy()


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def iter_video_frames(path: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield a bounded set of temporally uniform frames.

    Complexity for a long video is O(number_of_samples * seek/decode), not
    O(number_of_video_frames). No decoded frame list is retained in memory.
    """
    duration = probe_duration(path)
    limit = max_frames or settings.MAX_MEDIA_FRAMES
    count = min(sample_count(duration), limit)
    timestamps = even_sample_timestamps(duration, count)
    logger.info(
        "Media %s: duration=%.1fs, sampling=%d/%d configured frames",
        path.name, duration, len(timestamps), limit,
    )
    for ts in timestamps:
        frame = _decode_one_frame(path, ts)
        if frame is not None:
            yield frame


def iter_gif_frames(path: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    # FFmpeg treats GIFs as animated media; the same seek strategy avoids
    # materializing all GIF frames in Python memory.
    yield from iter_video_frames(path, max_frames=max_frames)
