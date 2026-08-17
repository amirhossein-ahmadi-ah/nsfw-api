#!/usr/bin/env python3
"""
Simple integration test script for the NSFW API.

Usage:
    python test_api.py ./files/test.jpg ./files/test.png \
        ./files/test.webp ./files/test.gif ./files/test.mp4

Any subset of files may be given; each is classified by its extension.
Only files under an ALLOWED_DIRS directory (see .env) will be accepted
by the API -- copy your test files there first.
"""
from __future__ import annotations

import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8001"


def test_health() -> bool:
    print("=== /health ===")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print(r.status_code, r.json())
    return r.status_code == 200


def test_info() -> bool:
    print("\n=== /info ===")
    r = requests.get(f"{BASE_URL}/info", timeout=5)
    print(r.status_code, r.json())
    return r.status_code == 200


def test_detect(path: str) -> None:
    print(f"\n=== /detect: {path} ===")
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/detect", json={"path": path}, timeout=120)
    except requests.RequestException as exc:
        print(f"REQUEST FAILED: {exc}")
        return
    elapsed = time.time() - start

    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.json()}")
        return

    data = r.json()
    print(f"media_type:           {data['media_type']}")
    print(f"checked_frames:       {data['checked_frames']}")
    print(f"suspicious_frames:    {data['suspicious_frames']}")
    print(f"suspicious_ratio:     {data['suspicious_ratio']}")
    print(f"nsfw:                 {data['nsfw']}")
    print(f"nsfl:                 {data['nsfl']}")
    print(f"sfw:                  {data['sfw']}")
    print(f"max_consecutive:      {data['max_consecutive_suspicious']}")
    print(f"decision_reason:      {data['decision_reason']}")
    print(f"is_nsfw:              {data['is_nsfw']}")
    print(f"is_nsfl:              {data['is_nsfl']}")
    print(f"processing_time_ms:   {data['processing_time_ms']} (wall: {elapsed*1000:.0f}ms)")


def main() -> int:
    if not test_health():
        print("Health check failed -- is the service running?")
        return 1
    test_info()

    paths = sys.argv[1:]
    if not paths:
        print(
            "\nNo test files given. Example:\n"
            "  python test_api.py ./files/test.jpg ./files/test.mp4"
        )
        return 0

    for p in paths:
        test_detect(p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
