"""Pydantic models for API requests and responses."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    path: str = Field(..., description="Absolute local filesystem path to the media file.")


class DetectResponse(BaseModel):
    success: bool = True
    path: str
    media_type: str  # "image" | "gif" | "video"
    is_nsfw: bool
    is_nsfl: bool
    nsfw: float
    nsfl: float
    sfw: float
    checked_frames: int
    suspicious_frames: int
    suspicious_ratio: float
    max_nsfw: float
    average_nsfw: float
    max_consecutive_suspicious: int
    average_suspicious_score: Optional[float] = None
    decision_reason: str
    processing_time_ms: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    api_version: str
    model_name: str
    model_loaded: bool
    onnx_providers: list
    nsfw_threshold: float
    nsfl_threshold: float
    min_suspicious_ratio: float
    min_consecutive_frames: int
    max_media_frames: int
    min_media_frames: int
    sample_interval_seconds: float
    inference_batch_size: int
    max_file_size_mb: int
    max_concurrent_inference: int
    detection_timeout: float
    supported_formats: dict


class StatsResponse(BaseModel):
    rss_memory_mb: float
    cpu_count: int
    model_loaded: bool
    active_inference_count: int
    queued_requests: int
    uptime_seconds: float
