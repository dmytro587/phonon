#!/usr/bin/env python3
"""POST the same WAV to a running fermion serve; print RSS after each request."""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        subprocess.check_call(
            ["kill", "-0", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def rss_kb(pid: int) -> int:
    if not process_alive(pid):
        return 0
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    if not out:
        return 0
    return int(out.split()[0])


def post(url: str, wav: str, hotwords: str | None) -> None:
    data = []
    boundary = "----fermionmem"
    with open(wav, "rb") as f:
        body = f.read()
    data.append(f"--{boundary}\r\n".encode())
    data.append(b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n')
    data.append(b"Content-Type: audio/wav\r\n\r\n")
    data.append(body)
    data.append(f"\r\n--{boundary}\r\n".encode())
    data.append(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    data.append(b"phonon-2\r\n")
    if hotwords:
        data.append(f"--{boundary}\r\n".encode())
        data.append(b'Content-Disposition: form-data; name="hotwords"\r\n\r\n')
        data.append(hotwords.encode())
        data.append(b"\r\n")
    data.append(f"--{boundary}--\r\n".encode())
    payload = b"".join(data)
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8765/v1/audio/transcriptions")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--hotwords", default="")
    args = p.parse_args()
    hw = args.hotwords.strip() or None
    print("idx\trss_kb", flush=True)
    for i in range(args.n):
        if not process_alive(args.pid):
            print(f"{i + 1}\t0", flush=True)
            print("error: fermion serve process is not running", file=sys.stderr)
            return 1
        try:
            post(args.url, args.wav, hw)
        except urllib.error.URLError as exc:
            print(f"{i + 1}\t{rss_kb(args.pid)}", flush=True)
            print(f"error: request failed: {exc}", file=sys.stderr)
            return 1
        print(f"{i + 1}\t{rss_kb(args.pid)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
