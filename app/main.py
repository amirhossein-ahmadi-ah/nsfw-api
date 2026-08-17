"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --host 127.0.0.1 --port 8001

Do NOT run multiple uvicorn workers -- the ONNX model is loaded once into
this process's memory at startup and must not be duplicated.
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.detector import (
    aggregate_scores,
    aggregate_single_frame,
    detector,
)
from app.media import classify_media_type, ffmpeg_binary, ffprobe_binary, iter_gif_frames, iter_video_frames, load_image_rgb
from app.schemas import (
    DetectRequest,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    StatsResponse,
)
from app.security import PathSecurityError, resolve_safe_path, validate_file

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("nsfw_api")
logger.setLevel(settings.LOG_LEVEL)

_formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

_file_handler = logging.handlers.RotatingFileHandler(
    settings.LOG_DIR / "nsfw-api.log",
    maxBytes=settings.LOG_MAX_BYTES,
    backupCount=settings.LOG_BACKUP_COUNT,
)
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

# Silence overly chatty third-party loggers a bit.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_start_time = time.monotonic()

app = FastAPI(
    title="NSFW/NSFL Local Classification API",
    version=__version__,
    description=(
        "Localhost-only media safety classification API. Accepts a local "
        "file path (image/gif/video) and returns NSFW/NSFL/SFW scores."
    ),
)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting NSFW API v%s", __version__)
    logger.info("Allowed directories: %s", settings.ALLOWED_DIRS)
    # Fail fast if video dependencies are missing. Images do not need FFmpeg,
    # but a healthy service must be able to serve every advertised media type.
    ffmpeg_binary()
    ffprobe_binary()
    try:
        detector.load()
        logger.info("Model loaded")
    except Exception:
        logger.exception("FATAL: model failed to load")
        raise


# ----------------------------------------------------------------------
# Error helpers
# ----------------------------------------------------------------------
_ERROR_STATUS = {
    "INVALID_PATH": 400,
    "SERVER_MISCONFIGURED": 500,
    "PATH_NOT_ALLOWED": 403,
    "FILE_NOT_FOUND": 404,
    "NOT_A_FILE": 400,
    "UNSUPPORTED_FORMAT": 400,
    "FILE_TOO_LARGE": 400,
    "EMPTY_FILE": 400,
    "CORRUPTED_MEDIA": 400,
    "MODEL_NOT_LOADED": 503,
    "DETECTION_TIMEOUT": 504,
    "INTERNAL_ERROR": 500,
}


def _error_response(code: str, message: str) -> JSONResponse:
    status_code = _ERROR_STATUS.get(code, 500)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=code, message=message).model_dump(),
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight liveness check. Never runs model inference."""
    return HealthResponse(status="ok", model_loaded=detector.loaded)


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    return InfoResponse(
        api_version=__version__,
        model_name=settings.MODEL_REPO,
        model_loaded=detector.loaded,
        onnx_providers=detector.providers,
        nsfw_threshold=settings.NSFW_THRESHOLD,
        nsfl_threshold=settings.NSFL_THRESHOLD,
        min_suspicious_ratio=settings.MIN_SUSPICIOUS_RATIO,
        min_consecutive_frames=settings.MIN_CONSECUTIVE_FRAMES,
        max_media_frames=settings.MAX_MEDIA_FRAMES,
        min_media_frames=settings.MIN_MEDIA_FRAMES,
        sample_interval_seconds=settings.SAMPLE_INTERVAL_SECONDS,
        inference_batch_size=settings.INFERENCE_BATCH_SIZE,
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        max_concurrent_inference=settings.MAX_CONCURRENT_INFERENCE,
        detection_timeout=settings.DETECTION_TIMEOUT,
        supported_formats={
            "image": sorted(settings.IMAGE_EXTENSIONS),
            "gif": sorted(settings.GIF_EXTENSIONS),
            "video": sorted(settings.VIDEO_EXTENSIONS),
        },
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    import os

    import psutil

    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    return StatsResponse(
        rss_memory_mb=round(rss_mb, 2),
        cpu_count=psutil.cpu_count() or 0,
        model_loaded=detector.loaded,
        active_inference_count=detector.active_inference_count,
        queued_requests=detector.queued_requests,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@app.post("/detect")
async def detect(payload: DetectRequest, request: Request):
    start = time.monotonic()

    # --- Security / validation --------------------------------------
    try:
        safe_path = resolve_safe_path(payload.path)
        ext, _size_bytes = validate_file(safe_path)
    except PathSecurityError as exc:
        logger.warning("Rejected path %s: %s (%s)", payload.path, exc.message, exc.code)
        return _error_response(exc.code, exc.message)

    if not detector.loaded:
        return _error_response("MODEL_NOT_LOADED", "Model is still loading, try again shortly.")

    media_type = classify_media_type(ext)
    logger.info("Detecting %s: %s", media_type, safe_path.name)

    try:
        result = await asyncio.wait_for(
            _run_detection(media_type, safe_path),
            timeout=settings.DETECTION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Detection timed out after %ss: %s", settings.DETECTION_TIMEOUT, safe_path)
        return _error_response(
            "DETECTION_TIMEOUT",
            f"Detection exceeded timeout of {settings.DETECTION_TIMEOUT}s.",
        )
    except ValueError as exc:
        # Raised by media.py for corrupted/unopenable files.
        logger.warning("Corrupted media %s: %s", safe_path, exc)
        return _error_response("CORRUPTED_MEDIA", str(exc))
    except Exception:
        logger.exception("Unexpected error detecting %s", safe_path)
        return _error_response("INTERNAL_ERROR", "An internal error occurred during detection.")

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "Result: %s nsfw=%.3f nsfl=%.3f sfw=%.3f frames=%d suspicious=%d (%dms)",
        "NSFW" if result.is_nsfw else ("NSFL" if result.is_nsfl else "SFW"),
        result.nsfw,
        result.nsfl,
        result.sfw,
        result.checked_frames,
        result.suspicious_frames,
        elapsed_ms,
    )

    response = DetectResponse(
        success=True,
        path=str(safe_path),
        media_type=media_type,
        is_nsfw=result.is_nsfw,
        is_nsfl=result.is_nsfl,
        nsfw=result.nsfw,
        nsfl=result.nsfl,
        sfw=result.sfw,
        checked_frames=result.checked_frames,
        suspicious_frames=result.suspicious_frames,
        suspicious_ratio=result.suspicious_ratio,
        max_nsfw=result.max_nsfw,
        average_nsfw=result.average_nsfw,
        max_consecutive_suspicious=result.max_consecutive_suspicious,
        average_suspicious_score=result.average_suspicious_score,
        decision_reason=result.decision_reason,
        processing_time_ms=elapsed_ms,
    )
    return response


async def _run_detection(media_type: str, path: Path):
    """Dispatch to the appropriate loader + scorer for the media type."""
    if media_type == "image":
        try:
            rgb = load_image_rgb(path)
        except Exception as exc:  # Pillow raises various errors for corrupt files
            raise ValueError(f"Could not read image: {exc}") from exc
        scores = await detector.score_frames([rgb])
        if not scores:
            raise ValueError("Image could not be scored (possibly corrupted).")
        return aggregate_single_frame(scores[0])

    if media_type == "gif":
        try:
            scores = await detector.score_frame_stream(
                iter_gif_frames(path, settings.MAX_MEDIA_FRAMES)
            )
        except Exception as exc:
            raise ValueError(f"Could not read GIF: {exc}") from exc
        return aggregate_scores(scores)

    if media_type == "video":
        try:
            scores = await detector.score_frame_stream(
                iter_video_frames(path, settings.MAX_MEDIA_FRAMES)
            )
        except Exception as exc:
            raise ValueError(f"Could not read video: {exc}") from exc
        return aggregate_scores(scores)

    raise ValueError(f"Unhandled media type: {media_type}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return _error_response("INTERNAL_ERROR", "An internal error occurred.")
