#!/usr/bin/env python3
"""Start fermion serve, loop transcriptions, print RSS; then stop serve."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from http_mem_loop import post, process_alive, rss_kb


def wait_health(url: str, timeout_s: float = 300.0) -> None:
    health = url.replace("/v1/audio/transcriptions", "/health")
    if not health.endswith("/health"):
        base = url.split("/v1/")[0]
        health = base + "/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    raise SystemExit(f"serve not healthy within {timeout_s}s ({health})")


def _drain(pipe, buf: list[str]) -> None:
    if pipe is None:
        return
    for line in pipe:
        buf.append(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--hotwords", default="")
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()

    fermion = shutil.which("fermion")
    if not fermion:
        raise SystemExit("fermion not on PATH")

    env = os.environ.copy()
    env.setdefault(
        "FERMION_CACHE_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache", "fermion")),
    )
    env.setdefault("FERMION_P2_FAST", "tdt16")
    env.setdefault("PHONON_ENC", "q8")

    cmd = [
        fermion,
        "serve",
        "phonon-2",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log: list[str] = []
    threading.Thread(target=_drain, args=(proc.stdout, log), daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/v1/audio/transcriptions"
    status = 0
    try:
        wait_health(url)
        hw = args.hotwords.strip() or None
        print("idx\trss_kb", flush=True)
        for i in range(args.n):
            if proc.poll() is not None:
                print(f"{i + 1}\t0", flush=True)
                print("error: fermion serve exited early", file=sys.stderr)
                status = 1
                break
            try:
                post(url, args.wav, hw)
            except urllib.error.URLError as exc:
                print(f"{i + 1}\t{rss_kb(proc.pid)}", flush=True)
                print(f"error: request failed: {exc}", file=sys.stderr)
                status = 1
                break
            rss = rss_kb(proc.pid)
            print(f"{i + 1}\t{rss}", flush=True)
            if rss == 0:
                print("error: serve process died during loop", file=sys.stderr)
                status = 1
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        time.sleep(0.5)

    if status != 0 and log:
        print("--- serve log (tail) ---", file=sys.stderr)
        for line in log[-40:]:
            print(line.rstrip(), file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
