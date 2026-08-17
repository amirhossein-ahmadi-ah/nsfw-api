#!/usr/bin/env python3
"""
Benchmark repeated /detect calls against a single file.

Usage:
    python benchmark.py ./files/test.jpg [num_runs]
"""
from __future__ import annotations

import statistics
import sys
import time

import psutil
import requests

BASE_URL = "http://127.0.0.1:8001"


def get_service_rss_mb() -> float:
    """Best-effort: find the uvicorn/nsfw-api process and read its RSS."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "uvicorn" in cmdline and "app.main:app" in cmdline:
                return proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Fallback: ask the API's own /stats endpoint.
    try:
        r = requests.get(f"{BASE_URL}/stats", timeout=5)
        if r.status_code == 200:
            return r.json()["rss_memory_mb"]
    except requests.RequestException:
        pass
    return -1.0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <path> [num_runs=10]")
        return 1

    path = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    ram_before = get_service_rss_mb()

    latencies = []
    for i in range(num_runs):
        start = time.time()
        r = requests.post(f"{BASE_URL}/detect", json={"path": path}, timeout=120)
        elapsed_ms = (time.time() - start) * 1000
        if r.status_code != 200:
            print(f"Run {i+1}: HTTP {r.status_code} - {r.json()}")
            continue
        latencies.append(elapsed_ms)
        print(f"Run {i+1}/{num_runs}: {elapsed_ms:.1f} ms")

    ram_after = get_service_rss_mb()

    if latencies:
        print("\n--- Results ---")
        print(f"Average latency: {statistics.mean(latencies):.1f} ms")
        print(f"Min latency:     {min(latencies):.1f} ms")
        print(f"Max latency:     {max(latencies):.1f} ms")
        if len(latencies) > 1:
            print(f"Std dev:         {statistics.stdev(latencies):.1f} ms")
    else:
        print("No successful runs.")

    print(f"\nRAM before: {ram_before:.1f} MB" if ram_before >= 0 else "\nRAM before: unavailable")
    print(f"RAM after:  {ram_after:.1f} MB" if ram_after >= 0 else "RAM after:  unavailable")

    return 0


if __name__ == "__main__":
    sys.exit(main())
