"""
Example async client for the Telegram bot process to call the local
NSFW/NSFL classification API.

This file lives outside the API's own venv -- copy the `detect_file`
function (and the httpx dependency) into the bot's codebase. The bot
already has its own Python 3.12 environment at:
    /root/.pyenv/versions/3.12.10/bin/python

Install in the BOT's environment (not the API's venv):
    pip install httpx

Usage from the bot:

    from nsfw_client import detect_file, NsfwApiUnavailable

    result = await detect_file("./files/abc.mp4")
    if result is None:
        # API was unavailable / timed out -- decide your bot's fallback
        # policy here (e.g. allow through, or hold for manual review).
        ...
    elif result["is_nsfw"] or result["is_nsfl"]:
        # block / warn / delete
        ...
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger("bot.nsfw_client")

NSFW_API_BASE_URL = "http://127.0.0.1:8001"
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=65.0, write=5.0, pool=5.0)


class NsfwApiUnavailable(Exception):
    """Raised when the API cannot be reached at all (connection refused, etc.)."""


async def detect_file(path: str) -> Optional[dict]:
    """
    Call the local NSFW/NSFL API for the given local file path.

    Returns the parsed JSON result dict on success, or None if the API
    could not be reached, timed out, or returned an error -- callers
    should treat None as "classification unavailable" and apply their
    own fallback policy rather than assuming the file is safe or unsafe.

    This function never raises for expected failure modes (connection
    errors, timeouts, non-2xx responses, malformed JSON); it logs and
    returns None so the Telegram bot never crashes because the
    classification service is temporarily down.
    """
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{NSFW_API_BASE_URL}/detect",
                json={"path": path},
            )
    except httpx.ConnectError:
        logger.error("NSFW API is unreachable (connection refused) for %s", path)
        return None
    except httpx.TimeoutException:
        logger.error("NSFW API request timed out for %s", path)
        return None
    except httpx.HTTPError as exc:
        logger.error("NSFW API request failed for %s: %s", path, exc)
        return None

    if response.status_code != 200:
        try:
            error_body = response.json()
        except ValueError:
            error_body = {"error": "UNKNOWN", "message": response.text[:200]}
        logger.warning(
            "NSFW API returned HTTP %s for %s: %s",
            response.status_code,
            path,
            error_body,
        )
        return None

    try:
        data = response.json()
    except ValueError:
        logger.error("NSFW API returned invalid JSON for %s", path)
        return None

    # Basic shape validation so callers can trust the keys they read.
    required_keys = {"success", "is_nsfw", "is_nsfl", "nsfw", "nsfl", "sfw", "media_type"}
    if not required_keys.issubset(data.keys()):
        logger.error("NSFW API response missing expected keys for %s: %s", path, data)
        return None

    if not data.get("success", False):
        logger.warning("NSFW API reported failure for %s: %s", path, data)
        return None

    return data


async def check_health() -> bool:
    """Optional: ping /health before relying on the API, e.g. at bot startup."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{NSFW_API_BASE_URL}/health")
        return response.status_code == 200 and response.json().get("model_loaded") is True
    except httpx.HTTPError:
        return False
