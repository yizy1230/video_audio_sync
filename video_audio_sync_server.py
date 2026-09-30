#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Audio Sync Server v1.0
FastAPI server wrapping the video audio sync analysis pipeline.

Routes:
    GET  /ready          -> Check if service is ready (200 = ok)
    GET  /test           -> Run predefined test cases and return results
    POST /video_audio_sync  -> Process a video audio sync request

Usage:
    python video_audio_sync_server.py --device cuda --port 8000
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
SYNCHFORMER_DIR = BASE_DIR / "Synchformer"
SYNCNET_DIR = BASE_DIR / "syncnet_python"
IMAGEBIND_DIR = BASE_DIR / "ImageBind"

sys.path.insert(0, str(SYNCHFORMER_DIR))
sys.path.insert(0, str(SYNCNET_DIR))
sys.path.insert(0, str(SYNCHFORMER_DIR / "model" / "modules" / "feat_extractors" / "visual"))
sys.path.insert(0, str(IMAGEBIND_DIR))

sys.path.insert(0, str(BASE_DIR))
from video_audio_sync import ModelManager, process_video

# ---------------------------------------------------------------------------
# App & Model Manager
# ---------------------------------------------------------------------------
app = FastAPI(title="Video Audio Sync Analysis v1.0", version="1.0.0")
model_mgr: Optional[ModelManager] = None
server_device: str = "cpu"
server_args: Optional[argparse.Namespace] = None


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class VideoAudioSyncRequest(BaseModel):
    video_path: str = Field(..., description="Local path or URL to the video file")
    sample_fps: Optional[float] = Field(2.0, description="Video sampling FPS for ImageBind")
    enable_semantic: Optional[bool] = Field(True, description="Enable ImageBind semantic alignment")
    enable_temporal: Optional[bool] = Field(True, description="Enable Synchformer temporal sync")
    enable_frame_level: Optional[bool] = Field(True, description="Enable SyncNet frame-level sync")
    sync_threshold: Optional[float] = Field(0.5, description="Overall sync score threshold")
    semantic_threshold: Optional[float] = Field(0.2, description="ImageBind cosine similarity threshold")
    temporal_offset_threshold: Optional[float] = Field(0.5, description="Synchformer offset threshold (seconds)")
    frame_level_threshold: Optional[float] = Field(5.0, description="SyncNet confidence threshold")
    min_duration: Optional[float] = Field(5.0, description="Minimum video duration in seconds for Synchformer")
    max_duration: Optional[float] = Field(60.0, description="Maximum video duration in seconds; longer videos will be truncated")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def init_model_manager(device: str, args: argparse.Namespace):
    global model_mgr, server_device, server_args
    if torch.cuda.is_available() and device == "cuda":
        print("Using GPU for inference.")
        server_device = "cuda"
    else:
        print("Using CPU for inference.")
        server_device = "cpu"
    print(f"[INIT] Loading models on device: {server_device} ...")
    model_mgr = ModelManager(device=server_device)

    # Prefetch weights then park on CPU; process_video moves one stage to GPU at a time.
    if args.enable_semantic:
        model_mgr.load_imagebind()
        model_mgr.release_imagebind()
    if args.enable_temporal:
        try:
            model_mgr.load_synchformer("24-01-04T16-39-21")
            model_mgr.release_synchformer()
        except Exception as exc:
            logger.warning("Synchformer load skipped: %s", exc)
    if args.enable_frame_level:
        try:
            model_mgr.load_syncnet()
            model_mgr.release_syncnet()
        except Exception as exc:
            logger.warning("SyncNet load skipped: %s", exc)
    print("[INIT] Model initialization complete (parked on CPU).")
    server_args = args


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/ready")
def ready():
    if model_mgr is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return {"status": "ok"}


@app.get("/test")
def test_endpoint():
    if model_mgr is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    test_dir = BASE_DIR / "test"
    test_files = sorted(test_dir.glob("test_*.json"))
    results = []
    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            req = json.load(f)
        try:
            res = process_video(
                video_path=req["video_path"],
                model_mgr=model_mgr,
                sample_fps=server_args.sample_fps if server_args else 2.0,
                enable_semantic=server_args.enable_semantic if server_args else True,
                enable_temporal=server_args.enable_temporal if server_args else True,
                enable_frame_level=server_args.enable_frame_level if server_args else True,
                semantic_threshold=server_args.semantic_threshold if server_args else 0.2,
                temporal_offset_threshold=server_args.temporal_offset_threshold if server_args else 0.5,
                frame_level_threshold=server_args.frame_level_threshold if server_args else 5.0,
                sync_threshold=server_args.sync_threshold if server_args else 0.5,
                min_duration=server_args.min_duration if server_args else 5.0,
                max_duration=server_args.max_duration if server_args else 60.0,
            )
            out = {
                "video_path": res["video_path"],
                "has_audio": res.get("has_audio", True),
                "semantic_alignment": res["semantic_alignment"],
                "temporal_sync": res["temporal_sync"],
                "frame_level_sync": res["frame_level_sync"],
                "overall_sync_score": res["overall_sync_score"],
                "overall_is_sync": res["overall_is_sync"],
                "timings": res.get("timings", {}),
                "algo_name": res["algo_name"],
                "algo_version": res["algo_version"],
                "source_test": str(tf.name),
            }
            results.append({"status": "success", "result": out})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({"status": "error", "source_test": str(tf.name), "error": str(exc)})
    return {"tests_run": len(results), "results": results}


@app.post("/video_audio_sync")
def video_audio_sync(req: VideoAudioSyncRequest):
    if model_mgr is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    semantic_threshold = req.semantic_threshold if req.semantic_threshold is not None else (server_args.semantic_threshold if server_args else 0.2)
    temporal_offset_threshold = req.temporal_offset_threshold if req.temporal_offset_threshold is not None else (server_args.temporal_offset_threshold if server_args else 0.5)
    frame_level_threshold = req.frame_level_threshold if req.frame_level_threshold is not None else (server_args.frame_level_threshold if server_args else 5.0)
    sync_threshold = req.sync_threshold if req.sync_threshold is not None else (server_args.sync_threshold if server_args else 0.5)

    # Models are loaded/released per stage inside process_video (GPU stage-serial).
    try:
        res = process_video(
            video_path=req.video_path,
            model_mgr=model_mgr,
            sample_fps=req.sample_fps or 2.0,
            enable_semantic=req.enable_semantic if req.enable_semantic is not None else True,
            enable_temporal=req.enable_temporal if req.enable_temporal is not None else True,
            enable_frame_level=req.enable_frame_level if req.enable_frame_level is not None else True,
            semantic_threshold=semantic_threshold,
            temporal_offset_threshold=temporal_offset_threshold,
            frame_level_threshold=frame_level_threshold,
            sync_threshold=sync_threshold,
            min_duration=req.min_duration if req.min_duration is not None else (server_args.min_duration if server_args else 5.0),
            max_duration=req.max_duration if req.max_duration is not None else (server_args.max_duration if server_args else 60.0),
        )
        print(f"[video_audio_sync] Timings (seconds): {res.get('timings', {})}")
        cuda_mem = {}
        if torch.cuda.is_available():
            cuda_mem = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }
            torch.cuda.empty_cache()
        return {
            "video_path": res["video_path"],
            "has_audio": res.get("has_audio", True),
            "semantic_alignment": res["semantic_alignment"],
            "temporal_sync": res["temporal_sync"],
            "frame_level_sync": res["frame_level_sync"],
            "overall_sync_score": res["overall_sync_score"],
            "overall_is_sync": res["overall_is_sync"],
            "timings": res.get("timings", {}),
            "algo_name": res["algo_name"],
            "algo_version": res["algo_version"],
            "cuda_mem": cuda_mem if cuda_mem else None,
        }
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}\n\n{tb}")


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Audio Sync Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Video sampling FPS for ImageBind")
    parser.add_argument("--enable-semantic", type=lambda x: x.lower() == "true", default=True, help="Enable ImageBind at startup")
    parser.add_argument("--enable-temporal", type=lambda x: x.lower() == "true", default=True, help="Enable Synchformer at startup")
    parser.add_argument("--enable-frame-level", type=lambda x: x.lower() == "true", default=True, help="Enable SyncNet at startup")
    parser.add_argument("--semantic-threshold", type=float, default=0.2, help="ImageBind cosine similarity threshold")
    parser.add_argument("--temporal-offset-threshold", type=float, default=0.5, help="Synchformer offset threshold (seconds)")
    parser.add_argument("--frame-level-threshold", type=float, default=5.0, help="SyncNet confidence threshold")
    parser.add_argument("--sync-threshold", type=float, default=0.5, help="Overall sync score threshold")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum video duration in seconds for Synchformer (default: 5)")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Maximum video duration in seconds; longer videos will be truncated (default: 60)")
    args, _ = parser.parse_known_args()

    init_model_manager(device=args.device, args=args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
