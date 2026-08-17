"""
ONNX model wrapper, single-image inference, and the aggregation/decision
logic that turns a set of per-frame scores into a final is_nsfw / is_nsfl
verdict for GIFs and videos.

The model is loaded exactly once (see `Detector.load()`, called at FastAPI
startup) and reused for every request. Inference concurrency is bounded by
an asyncio.Semaphore configured via MAX_CONCURRENT_INFERENCE.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("nsfw_api.detector")

# Model output index order, per spec: [NSFL, NSFW, SFW]
IDX_NSFL = 0
IDX_NSFW = 1
IDX_SFW = 2


@dataclass
class FrameScore:
    nsfl: float
    nsfw: float
    sfw: float


@dataclass
class AggregateResult:
    checked_frames: int
    suspicious_frames: int
    suspicious_ratio: float
    max_nsfw: float
    average_nsfw: float
    max_consecutive_suspicious: int
    average_suspicious_score: Optional[float]
    nsfw: float
    nsfl: float
    sfw: float
    is_nsfw: bool
    is_nsfl: bool
    decision_reason: str


class ModelNotLoadedError(RuntimeError):
    pass


class Detector:
    """Owns the ONNX Runtime session. One instance per process."""

    def __init__(self) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._input_size = settings.MODEL_INPUT_SIZE
        self._loaded = False
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)
        self._active_inference_count = 0
        self._queued_requests = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the ONNX model once. Call at application startup."""
        import onnxruntime as ort

        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {settings.MODEL_PATH}. "
                f"Run download_model.py first."
            )

        so = ort.SessionOptions()
        # Keep intra-op threading modest -- we only have 4 cores and the
        # Telegram bot needs headroom too.
        so.intra_op_num_threads = max(1, min(4, settings.MAX_CONCURRENT_INFERENCE * 2))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(settings.MODEL_PATH),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Warmup inference so the first real request isn't the slow one.
        dummy = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
        try:
            self._session.run(None, {self._input_name: dummy})
        except Exception as exc:  # noqa: BLE001 - warmup best-effort
            logger.warning("Model warmup inference failed (non-fatal): %s", exc)

        self._loaded = True
        logger.info(
            "Model loaded from %s (providers=%s)",
            settings.MODEL_PATH,
            self._session.get_providers(),
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def providers(self) -> List[str]:
        if not self._session:
            return []
        return self._session.get_providers()

    @property
    def active_inference_count(self) -> int:
        return self._active_inference_count

    @property
    def queued_requests(self) -> int:
        return self._queued_requests

    # ------------------------------------------------------------------
    # Preprocessing / raw inference
    # ------------------------------------------------------------------
    def _preprocess(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Resize to the model's expected input and convert to NCHW float32.

        IMPORTANT: this model expects RAW pixel values in [0, 255] as
        float32 -- NOT divided by 255, NOT mean/std normalized. This
        matches the reference implementation confirmed to work correctly
        against this exact model. Do not add a /255.0 or [-1,1] rescale
        here; that silently produces near-uniform low-confidence outputs
        and causes false negatives (real NSFW/NSFL content scored as SFW).
        """
        if rgb_array.shape[:2] == (self._input_size, self._input_size):
            arr = np.asarray(rgb_array, dtype=np.float32)
        else:
            from PIL import Image
            img = Image.fromarray(rgb_array).resize(
                (self._input_size, self._input_size), Image.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def _run_batch(self, frames: List[np.ndarray]) -> List[FrameScore]:
        if not frames:
            return []
        if not self._loaded or self._session is None:
            raise ModelNotLoadedError("Model is not loaded.")
        tensors = [self._preprocess(frame)[0] for frame in frames]
        batch = np.stack(tensors, axis=0).astype(np.float32, copy=False)
        try:
            outputs = self._session.run(None, {self._input_name: batch})
        except Exception:
            # Some ONNX exports hard-code batch=1. Keep batching as a fast
            # path but automatically fall back to single-frame inference so
            # compatibility is never sacrificed.
            if len(frames) == 1:
                raise
            return [self._run_single(frame) for frame in frames]
        probs = np.asarray(outputs[0])
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        results: List[FrameScore] = []
        for row in probs:
            row = np.asarray(row, dtype=np.float32).reshape(-1)
            if row.min() < 0 or row.max() > 1 or not np.isclose(row.sum(), 1.0, atol=0.05):
                exp = np.exp(row - np.max(row))
                row = exp / exp.sum()
            results.append(FrameScore(float(row[IDX_NSFL]), float(row[IDX_NSFW]), float(row[IDX_SFW])))
        return results

    def _run_single(self, rgb_array: np.ndarray) -> FrameScore:
        if not self._loaded or self._session is None:
            raise ModelNotLoadedError("Model is not loaded.")

        tensor = self._preprocess(rgb_array)
        outputs = self._session.run(None, {self._input_name: tensor})
        probs = np.asarray(outputs[0]).reshape(-1)

        # Defensive: some exported models emit logits rather than softmax
        # probabilities. If values fall outside [0, 1] or don't sum near 1,
        # apply softmax ourselves.
        if probs.min() < 0 or probs.max() > 1 or not np.isclose(probs.sum(), 1.0, atol=0.05):
            exp = np.exp(probs - np.max(probs))
            probs = exp / exp.sum()

        return FrameScore(
            nsfl=float(probs[IDX_NSFL]),
            nsfw=float(probs[IDX_NSFW]),
            sfw=float(probs[IDX_SFW]),
        )

    # ------------------------------------------------------------------
    # Public async entry points (bounded concurrency)
    # ------------------------------------------------------------------
    async def score_frame_stream(self, frames: Iterable[np.ndarray]) -> List[FrameScore]:
        """Score a lazy frame iterator without materializing video frames."""
        self._queued_requests += 1
        async with self._semaphore:
            self._queued_requests -= 1
            self._active_inference_count += 1
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._score_frame_stream_sync, frames)
            finally:
                self._active_inference_count -= 1

    def _score_frame_stream_sync(self, frames: Iterable[np.ndarray]) -> List[FrameScore]:
        scores: List[FrameScore] = []
        batch: List[np.ndarray] = []
        batch_size = max(1, settings.INFERENCE_BATCH_SIZE)
        for frame in frames:
            batch.append(frame)
            if len(batch) >= batch_size:
                try:
                    scores.extend(self._run_batch(batch))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Batch inference failed, skipping batch: %s", exc)
                batch.clear()
        if batch:
            try:
                scores.extend(self._run_batch(batch))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batch inference failed, skipping final batch: %s", exc)
        return scores

    async def score_frames(self, frames: List[np.ndarray]) -> List[FrameScore]:
        """
        Run inference on a list of already-sampled RGB frames, respecting
        the global concurrency semaphore. Frames are scored sequentially
        within one acquired slot (the model itself isn't thread-parallel
        here) but off the event loop via a thread executor.
        """
        self._queued_requests += 1
        async with self._semaphore:
            self._queued_requests -= 1
            self._active_inference_count += 1
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._score_frames_sync, frames)
            finally:
                self._active_inference_count -= 1

    def _score_frames_sync(self, frames: List[np.ndarray]) -> List[FrameScore]:
        scores = []
        for frame in frames:
            try:
                scores.append(self._run_single(frame))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Frame inference failed, skipping frame: %s", exc)
        return scores


def aggregate_scores(scores: List[FrameScore]) -> AggregateResult:
    """
    Turn a list of per-frame scores into a final decision.

    This deliberately avoids the naive `max_nsfw >= threshold` rule. It
    requires either:
      (a) a sufficiently high RATIO of suspicious frames, AND/OR
      (b) a run of MIN_CONSECUTIVE_FRAMES consecutive suspicious frames,
      (c) OR a single frame so extreme it exceeds SINGLE_FRAME_OVERRIDE
          (a safety valve for very short/rare but unambiguous content).
    NSFL and NSFW are evaluated independently using their own thresholds.
    """
    if not scores:
        return AggregateResult(
            checked_frames=0,
            suspicious_frames=0,
            suspicious_ratio=0.0,
            max_nsfw=0.0,
            average_nsfw=0.0,
            max_consecutive_suspicious=0,
            average_suspicious_score=None,
            nsfw=0.0,
            nsfl=0.0,
            sfw=1.0,
            is_nsfw=False,
            is_nsfl=False,
            decision_reason="NO_FRAMES_ANALYZED",
        )

    n = len(scores)
    nsfw_vals = [s.nsfw for s in scores]
    nsfl_vals = [s.nsfl for s in scores]
    sfw_vals = [s.sfw for s in scores]

    def _decide(
        vals: List[float],
        threshold: float,
        min_ratio: float,
        min_consecutive: int,
        min_absolute_count: int,
    ):
        suspicious_flags = [v >= threshold for v in vals]
        suspicious_count = sum(suspicious_flags)
        ratio = suspicious_count / n

        max_consecutive = 0
        current = 0
        for flag in suspicious_flags:
            if flag:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        max_val = max(vals)
        avg_val = sum(vals) / n
        suspicious_scores = [v for v, f in zip(vals, suspicious_flags) if f]
        avg_suspicious = (
            sum(suspicious_scores) / len(suspicious_scores) if suspicious_scores else None
        )

        single_frame_hit = max_val >= settings.SINGLE_FRAME_OVERRIDE
        # Ratio-based flagging requires BOTH a sufficient ratio AND a
        # sufficient absolute count of suspicious frames. Without the
        # absolute-count floor, a tiny sample (e.g. 1 of 5 frames = 20%)
        # would trip the ratio rule on a single isolated spike -- exactly
        # the false-positive pattern this system is designed to avoid.
        ratio_hit = ratio >= min_ratio and suspicious_count >= min_absolute_count
        consecutive_hit = max_consecutive >= min_consecutive

        # A sustained consecutive run is independently sufficient evidence
        # (temporally persistent content), as is a strong ratio backed by
        # enough absolute frames. A single isolated spike with neither is
        # NOT enough unless it clears the single-frame override ceiling.
        flagged = single_frame_hit or ratio_hit or consecutive_hit

        if single_frame_hit:
            reason = "SINGLE_FRAME_EXTREME_SCORE"
        elif ratio_hit and consecutive_hit:
            reason = "SUSPICIOUS_RATIO_AND_CONSECUTIVE_RUN"
        elif ratio_hit:
            reason = "SUSPICIOUS_RATIO_EXCEEDED"
        elif consecutive_hit:
            reason = "CONSECUTIVE_SUSPICIOUS_RUN"
        else:
            reason = "BELOW_THRESHOLD"

        return flagged, reason, suspicious_count, ratio, max_consecutive, max_val, avg_val, avg_suspicious

    (
        nsfw_flagged,
        nsfw_reason,
        nsfw_suspicious_count,
        nsfw_ratio,
        nsfw_max_consec,
        max_nsfw,
        avg_nsfw,
        avg_suspicious_nsfw,
    ) = _decide(
        nsfw_vals,
        settings.NSFW_THRESHOLD,
        settings.MIN_SUSPICIOUS_RATIO,
        settings.MIN_CONSECUTIVE_FRAMES,
        settings.MIN_SUSPICIOUS_ABSOLUTE_COUNT,
    )

    (
        nsfl_flagged,
        nsfl_reason,
        nsfl_suspicious_count,
        nsfl_ratio,
        nsfl_max_consec,
        max_nsfl,
        avg_nsfl,
        _avg_suspicious_nsfl,
    ) = _decide(
        nsfl_vals,
        settings.NSFL_THRESHOLD,
        settings.MIN_NSFL_SUSPICIOUS_RATIO,
        settings.MIN_NSFL_CONSECUTIVE_FRAMES,
        settings.MIN_NSFL_SUSPICIOUS_ABSOLUTE_COUNT,
    )

    avg_sfw = sum(sfw_vals) / n

    # NSFL and NSFW are reported independently, per spec -- NSFL must never
    # be folded into the NSFW verdict or vice versa.
    combined_reason = f"nsfw={nsfw_reason};nsfl={nsfl_reason}"

    return AggregateResult(
        checked_frames=n,
        suspicious_frames=nsfw_suspicious_count,
        suspicious_ratio=round(nsfw_ratio, 4),
        max_nsfw=round(max_nsfw, 4),
        average_nsfw=round(avg_nsfw, 4),
        max_consecutive_suspicious=nsfw_max_consec,
        average_suspicious_score=(
            round(avg_suspicious_nsfw, 4) if avg_suspicious_nsfw is not None else None
        ),
        nsfw=round(max_nsfw, 4),
        nsfl=round(max_nsfl, 4),
        sfw=round(avg_sfw, 4),
        is_nsfw=nsfw_flagged,
        is_nsfl=nsfl_flagged,
        decision_reason=combined_reason,
    )


def aggregate_single_frame(score: FrameScore) -> AggregateResult:
    """For plain still images: one frame, direct threshold comparison."""
    is_nsfw = score.nsfw >= settings.NSFW_THRESHOLD
    is_nsfl = score.nsfl >= settings.NSFL_THRESHOLD
    return AggregateResult(
        checked_frames=1,
        suspicious_frames=1 if is_nsfw else 0,
        suspicious_ratio=1.0 if is_nsfw else 0.0,
        max_nsfw=round(score.nsfw, 4),
        average_nsfw=round(score.nsfw, 4),
        max_consecutive_suspicious=1 if is_nsfw else 0,
        average_suspicious_score=round(score.nsfw, 4) if is_nsfw else None,
        nsfw=round(score.nsfw, 4),
        nsfl=round(score.nsfl, 4),
        sfw=round(score.sfw, 4),
        is_nsfw=is_nsfw,
        is_nsfl=is_nsfl,
        decision_reason="SINGLE_IMAGE_THRESHOLD",
    )


# Module-level singleton, imported by main.py
detector = Detector()
