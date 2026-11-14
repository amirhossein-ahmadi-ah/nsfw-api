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
        "-show_entries", "format=duration:stream=duration",
        "-print_format", "json",
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
    """Return duration without decoding media frames. Returns 0.0 if not determinable."""
    try:
        duration = _duration_seconds(_run_probe(path))
        return max(0.0, duration)
    except Exception as exc:
        logger.warning("Could not probe duration for %s: %s", path.name, exc)
        return 0.0


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
    cmd = [
        ffmpeg_binary(),
        "-y",
        "-noautorotate",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", os.fspath(path),
        "-an", "-sn", "-dn",
        "-threads", "2",
        "-vframes", "1",
        "-vf", f"scale={settings.MODEL_INPUT_SIZE}:{settings.MODEL_INPUT_SIZE}:flags=bilinear",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=settings.FFMPEG_FRAME_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.debug("FFmpeg seek timed out at %.3fs in %s", timestamp, path.name)
            return None
    except Exception as exc:
        logger.debug("FFmpeg frame seek failed at %.3fs: %s", timestamp, exc)
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
    if duration > 0:
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
    else:
        logger.info(
            "Media %s: duration unknown, using incremental sampling up to %d frames",
            path.name, limit,
        )
        ts = 0.0
        step = max(2.0, settings.SAMPLE_INTERVAL_SECONDS)
        yielded = 0
        consecutive_misses = 0
        while yielded < limit and consecutive_misses < 3:
            frame = _decode_one_frame(path, ts)
            if frame is not None:
                yield frame
                yielded += 1
                consecutive_misses = 0
            else:
                consecutive_misses += 1
            ts += step


def iter_gif_frames(path: Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    # FFmpeg treats GIFs as animated media; the same seek strategy avoids
    # materializing all GIF frames in Python memory.
    yield from iter_video_frames(path, max_frames=max_frames)
