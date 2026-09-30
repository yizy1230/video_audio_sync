#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Audio Sync Web UI (Gradio)

Features:
  - Sample / Upload / URL input
  - Sample mode loads precomputed test/result_local_*.json (no API re-run)
  - Call the /video_audio_sync API server for Upload/URL
  - Display sync analysis results (overall score, semantic alignment,
    temporal sync, frame-level lip sync, raw JSON)

Usage:
  # 1. Start the API server first (optional for Sample cached mode)
     python video_audio_sync_server.py --device cuda --port 8000
  # 2. Launch the Web UI
     python video_audio_sync_webui.py --server-url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Gradio import
# ---------------------------------------------------------------------------
try:
    import gradio as gr
except ImportError:
    print("[ERROR] gradio is not installed. Please run: pip install gradio")
    sys.exit(1)

BASE_DIR = Path(__file__).parent.resolve()
TEST_DIR = BASE_DIR / "test"
SAMPLE_DIR = TEST_DIR / "input_videos"
UPLOAD_DIR = BASE_DIR / "webui_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_SERVER_URL = "http://127.0.0.1:8001"


# ---------------------------------------------------------------------------
# Sample cache helpers
# ---------------------------------------------------------------------------
def _build_sample_result_map() -> dict:
    """Map sample video basename / abs path -> precomputed result_local_*.json."""
    mapping = {}
    for test_json in sorted(TEST_DIR.glob("test_local_*.json")):
        result_json = TEST_DIR / test_json.name.replace("test_local_", "result_local_", 1)
        if not result_json.is_file():
            continue
        try:
            with open(test_json, "r", encoding="utf-8") as f:
                inp = json.load(f)
            with open(result_json, "r", encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rel = inp.get("video_path") or result.get("video_path") or ""
        if not rel:
            continue
        abs_path = str((BASE_DIR / rel).resolve())
        mapping[os.path.basename(rel)] = result
        mapping[abs_path] = result
        mapping[rel] = result
    return mapping


def _load_cached_sample_result(video_path: str) -> Optional[Dict]:
    if not video_path:
        return None
    mapping = _build_sample_result_map()
    keys = [
        video_path,
        str(Path(video_path).resolve()) if os.path.exists(video_path) else None,
        os.path.basename(video_path),
    ]
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    return None


def _get_sample_videos() -> List[Tuple[str, str]]:
    cached_samples = []
    other_samples = []
    cached = _build_sample_result_map()
    if SAMPLE_DIR.is_dir():
        for f in sorted(SAMPLE_DIR.iterdir()):
            if f.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                size_mb = f.stat().st_size / (1024 * 1024)
                has_cache = f.name in cached or str(f.resolve()) in cached
                item = (f"{f.name} ({size_mb:.1f} MB){' [cached]' if has_cache else ''}", str(f))
                (cached_samples if has_cache else other_samples).append(item)
    return cached_samples + other_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def call_server_api(video_path: str, server_url: str) -> Optional[Dict]:
    """POST /video_audio_sync to the FastAPI server."""
    url = f"{server_url.rstrip('/')}/video_audio_sync"
    payload = {"video_path": video_path}
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"_error": f"Server API call failed: {exc}"}


def flatten_result(result: Dict) -> tuple:
    """
    Convert raw JSON result into display DataFrames and strings.
    Returns: (overall_score, overall_is_sync, has_audio, semantic_df,
              temporal_df, frame_df, raw_json, status_msg)
    """
    raw_json = json.dumps(result, indent=2, ensure_ascii=False)

    overall_score = result.get("overall_sync_score", 0.0)
    overall_is_sync = result.get("overall_is_sync", False)
    has_audio = result.get("has_audio", True)

    # When video has no audio, show clear "not applicable" in all tables
    if not has_audio:
        semantic_df = pd.DataFrame([{
            "Metric": "Semantic alignment (ImageBind)",
            "Cosine similarity": "—",
            "In sync": "⚠️ No audio stream; analysis skipped",
        }])
        temporal_df = pd.DataFrame([{
            "Metric": "Temporal sync (Synchformer)",
            "Offset (s)": "—",
            "Confidence": "—",
            "In sync": "⚠️ No audio stream; analysis skipped",
        }])
        frame_df = pd.DataFrame([{
            "Metric": "Frame-level lip sync (SyncNet)",
            "AV offset (frames)": "—",
            "Min distance": "—",
            "Confidence": "—",
            "In sync": "⚠️ No audio stream; analysis skipped",
        }])
        status_msg = "⚠️ This video has no audio stream. Audio-visual sync does not apply (every metric stays at its default)."
        return (
            overall_score,
            overall_is_sync,
            has_audio,
            semantic_df,
            temporal_df,
            frame_df,
            raw_json,
            status_msg,
        )

    # --- Semantic alignment ---
    semantic = result.get("semantic_alignment", {})
    if semantic:
        semantic_rows = [{
            "Metric": "Semantic alignment (ImageBind)",
            "Cosine similarity": semantic.get("cosine_similarity", "N/A"),
            "In sync": "✅ Yes" if semantic.get("is_semantic_sync") else "❌ No",
        }]
    else:
        semantic_rows = [{"Metric": "Semantic alignment", "Cosine similarity": "—", "In sync": "Disabled"}]
    semantic_df = pd.DataFrame(semantic_rows)

    # --- Temporal sync ---
    temporal = result.get("temporal_sync", {})
    if temporal:
        if "error" in temporal:
            temporal_rows = [{
                "Metric": "Temporal sync (Synchformer)",
                "Offset (s)": temporal.get("offset_seconds", temporal.get("offset_ms", "N/A")),
                "Confidence": temporal.get("confidence", "N/A"),
                "In sync": f"Inference failed: {temporal['error']}",
            }]
        else:
            temporal_rows = [{
                "Metric": "Temporal sync (Synchformer)",
                "Offset (s)": temporal.get("offset_seconds", "N/A"),
                "Confidence": round(temporal.get("confidence", 0.0), 4) if temporal.get("confidence") is not None else "N/A",
                "In sync": "✅ Yes" if temporal.get("is_temporal_sync") else "❌ No",
            }]
    else:
        temporal_rows = [{"Metric": "Temporal sync", "Offset (s)": "—", "Confidence": "—", "In sync": "Disabled"}]
    temporal_df = pd.DataFrame(temporal_rows)

    # --- Frame-level sync ---
    frame = result.get("frame_level_sync", {})
    if frame:
        if "error" in frame:
            frame_rows = [{
                "Metric": "Frame-level lip sync (SyncNet)",
                "AV offset (frames)": frame.get("av_offset_frames", "N/A"),
                "Min distance": frame.get("min_distance", "N/A"),
                "Confidence": frame.get("confidence", "N/A"),
                "In sync": f"Inference failed: {frame['error']}",
            }]
        else:
            frame_rows = [{
                "Metric": "Frame-level lip sync (SyncNet)",
                "AV offset (frames)": frame.get("av_offset_frames", "N/A"),
                "Min distance": round(frame.get("min_distance", 0.0), 4) if frame.get("min_distance") is not None else "N/A",
                "Confidence": round(frame.get("confidence", 0.0), 4) if frame.get("confidence") is not None else "N/A",
                "In sync": "✅ Yes" if frame.get("is_lip_sync") else "❌ No",
            }]
    else:
        frame_rows = [{"Metric": "Frame-level lip sync", "AV offset (frames)": "—", "Min distance": "—", "Confidence": "—", "In sync": "Disabled"}]
    frame_df = pd.DataFrame(frame_rows)

    # Status message
    if overall_is_sync:
        status_msg = f"✅ Audio and video are in sync (overall score: {overall_score:.4f})"
    else:
        status_msg = f"❌ Audio and video are out of sync (overall score: {overall_score:.4f})"

    return (
        overall_score,
        overall_is_sync,
        has_audio,
        semantic_df,
        temporal_df,
        frame_df,
        raw_json,
        status_msg,
    )


def _empty_outputs(status_msg: str):
    return (
        None, "—", "—",
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        "", status_msg,
    )


def _result_to_outputs(result: Dict, from_cache: bool = False):
    if "_error" in result:
        return _empty_outputs(f"❌ {result['_error']}")

    (
        score, is_sync, audio,
        semantic_df, temporal_df, frame_df,
        raw_json, status_msg,
    ) = flatten_result(result)

    audio_str = "✅ Has audio" if audio else "❌ No audio"
    sync_str = "✅ Yes" if is_sync else "❌ No"
    if from_cache:
        status_msg = f"📦 Loaded a cached sample result (API skipped) | {status_msg}"

    return (
        round(score, 4) if isinstance(score, (int, float)) else score,
        sync_str,
        audio_str,
        semantic_df,
        temporal_df,
        frame_df,
        raw_json,
        status_msg,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui(server_url: str) -> gr.Blocks:
    with gr.Blocks(title="🎬 Audio-visual sync v1.0", theme="default") as demo:
        gr.Markdown("# 🎬 Audio-visual sync v1.0")
        gr.Markdown(
            "Check whether the picture and the audio are in sync (semantic alignment / temporal sync / frame-level lip sync).\n\n"
            "**Sample mode**: a sample marked `[cached]` shows `test/result_local_*.json` directly and does not run inference again."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_mode = gr.Radio(
                    choices=["Sample", "Upload", "URL"],
                    value="Sample",
                    label="Input mode",
                    info="Sample: built-in clip (reads cache) | Upload: upload a video | URL: video link",
                )
                _samples = _get_sample_videos()
                sample_selector = gr.Dropdown(
                    choices=_samples,
                    value=_samples[0][1] if _samples else None,
                    label="Sample video ([cached] = stored result)",
                    visible=True,
                )
                video_input = gr.Video(
                    label="Upload video",
                    sources=["upload"],
                    visible=False,
                )
                url_input = gr.Textbox(
                    label="Video URL",
                    placeholder="https://example.com/video.mp4",
                    visible=False,
                )
                server_url_input = gr.Textbox(
                    label="Server URL",
                    value=server_url,
                    placeholder="http://127.0.0.1:8001",
                    info="Required for Upload and URL. Sample cache mode does not need the server.",
                )
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=2):
                status_text = gr.Textbox(label="Status", value="Waiting for a sample...", interactive=False)
                overall_score = gr.Number(label="Overall sync score", interactive=False)
                overall_is_sync = gr.Textbox(label="In sync", interactive=False)
                has_audio = gr.Textbox(label="Has audio", interactive=False)
                video_preview = gr.Video(label="Video preview", interactive=False)

        with gr.Tabs():
            with gr.TabItem("📊 Semantic alignment"):
                semantic_table = gr.Dataframe(
                    label="Semantic alignment (ImageBind)",
                    interactive=False,
                )
            with gr.TabItem("⏱️ Temporal sync"):
                temporal_table = gr.Dataframe(
                    label="Temporal sync (Synchformer)",
                    interactive=False,
                )
            with gr.TabItem("👄 Frame-level lip sync"):
                frame_table = gr.Dataframe(
                    label="Frame-level lip sync (SyncNet)",
                    interactive=False,
                )
            with gr.TabItem("📄 Raw JSON"):
                json_output = gr.Code(
                    label="Raw JSON",
                    language="json",
                    interactive=False,
                )

        result_outputs = [
            overall_score,
            overall_is_sync,
            has_audio,
            semantic_table,
            temporal_table,
            frame_table,
            json_output,
            status_text,
        ]

        def on_input_mode_change(mode):
            if mode == "Sample":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif mode == "Upload":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        input_mode.change(
            fn=on_input_mode_change,
            inputs=[input_mode],
            outputs=[sample_selector, video_input, url_input],
        )

        def on_sample_select(sample_video):
            if not sample_video or not os.path.exists(sample_video):
                return (None,) + _empty_outputs("❌ Select a sample video")
            cached = _load_cached_sample_result(sample_video)
            if cached is None:
                return (
                    sample_video,
                    None, "—", "—",
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    "",
                    "Sample selected (no cached result). Click Analyze to call the API.",
                )
            outs = _result_to_outputs(cached, from_cache=True)
            return (sample_video,) + outs

        def on_submit(input_mode_val, sample_video, video_path, video_url, srv_url):
            if input_mode_val == "Sample":
                if not sample_video or not os.path.exists(sample_video):
                    return (None,) + _empty_outputs("❌ Select a sample video")
                cached = _load_cached_sample_result(sample_video)
                if cached is not None:
                    outs = _result_to_outputs(cached, from_cache=True)
                    return (sample_video,) + outs
                video_path = sample_video
            elif input_mode_val == "Upload":
                if not video_path or not os.path.exists(video_path):
                    return (None,) + _empty_outputs("❌ Upload a video file first")
            elif input_mode_val == "URL":
                if not video_url or not video_url.strip().startswith(("http://", "https://")):
                    return (None,) + _empty_outputs("❌ Enter a valid http(s) URL")
                video_path = video_url.strip()
            else:
                return (None,) + _empty_outputs("❌ Unknown input mode")

            result = call_server_api(video_path, srv_url)
            outs = _result_to_outputs(result, from_cache=False)
            preview = video_path if os.path.exists(str(video_path)) else None
            return (preview,) + outs

        all_outputs = [video_preview] + result_outputs

        sample_selector.change(
            fn=on_sample_select,
            inputs=[sample_selector],
            outputs=all_outputs,
        )
        demo.load(
            fn=on_sample_select,
            inputs=[sample_selector],
            outputs=all_outputs,
        )
        submit_btn.click(
            fn=on_submit,
            inputs=[input_mode, sample_selector, video_input, url_input, server_url_input],
            outputs=all_outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Audio Sync Web UI")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"FastAPI server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Gradio server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio server port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Video Audio Sync Web UI")
    print("=" * 60)
    print(f"Server URL : {args.server_url}")
    print(f"Gradio host : {args.host}")
    print(f"Gradio port : {args.port}")
    print("=" * 60)

    demo = build_ui(server_url=args.server_url)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(SAMPLE_DIR)],
    )


if __name__ == "__main__":
    main()
