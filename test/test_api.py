#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end API test for video_audio_sync_server.

Usage (run from algos/video_audio_sync/v1_0/):
    python3 test/test_api.py --input-json test/test_local_1.json --output-dir test/server_results

Steps:
    1. Read input JSON.
    2. Spawn video_audio_sync_server.py as a subprocess.
    3. Poll GET /ready until the service is healthy.
    4. GET /test and save the result to <output-dir>/test_result.json.
    5. POST /video_audio_sync with the input payload.
    6. Save POST response JSON to <output-dir>/result.json.
    7. Terminate the server.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def wait_for_ready(url: str, timeout: float = 300.0, interval: float = 0.5) -> bool:
    """Poll the given URL until it returns HTTP 200 or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(interval)
    return False


def _http_get(url: str, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, data: dict, timeout: float = 600) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Test video_audio_sync_server end-to-end")
    parser.add_argument("--input-json", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device for server (cpu or cuda)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read input
    # ------------------------------------------------------------------
    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Start the server
    # ------------------------------------------------------------------
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "video_audio_sync_server.py",
    )
    print(f"[INFO] Starting server: {server_script}")
    proc = subprocess.Popen(
        [sys.executable, server_script, "--device", args.device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # --------------------------------------------------------------
        # 3. Wait for ready
        # --------------------------------------------------------------
        ready_url = "http://127.0.0.1:8000/ready"
        print(f"[INFO] Waiting for server at {ready_url} ...")
        if not wait_for_ready(ready_url, timeout=300):
            raise RuntimeError("Server failed to become ready within timeout.")

        print("[INFO] Server is ready.")

        input_stem = Path(args.input_json).stem

        # --------------------------------------------------------------
        # 4. GET /test and save
        # --------------------------------------------------------------
        print("[INFO] Sending GET /test ...")
        test_result = _http_get("http://127.0.0.1:8000/test", timeout=600)
        test_result_path = os.path.join(args.output_dir, f"{input_stem}_test_result.json")
        with open(test_result_path, "w", encoding="utf-8") as f:
            json.dump(test_result, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved /test result to {test_result_path}")

        # --------------------------------------------------------------
        # 5. POST /video_audio_sync
        # --------------------------------------------------------------
        print("[INFO] Sending POST /video_audio_sync ...")
        result = _http_post("http://127.0.0.1:8000/video_audio_sync", payload, timeout=600)

        # --------------------------------------------------------------
        # 6. Save JSON result
        # --------------------------------------------------------------
        result_path = os.path.join(args.output_dir, f"{input_stem}_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved video_audio_sync result to {result_path}")

    finally:
        print("[INFO] Terminating server ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
