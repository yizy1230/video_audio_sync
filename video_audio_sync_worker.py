#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery worker for video_audio_sync.

Consumes the ``video_audio_sync`` queue. Prefetches ImageBind / Synchformer /
SyncNet at startup, parks weights on CPU, and moves one stage to GPU at a time.

Start command:
    celery -A video_audio_sync_worker.app worker -Q video_audio_sync -c 1 -n video_audio_sync@%h
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).parent.resolve()
ALGOS_ROOT = SCRIPT_DIR.parent.parent.resolve()
if str(ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGOS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("WORKER_CONFIG_PATH", str(SCRIPT_DIR / "worker_config.yaml"))

import time

import torch
from celery.signals import worker_process_init
from loguru import logger

from common.worker_base import create_worker_app, execute_algo_task  # noqa: E402
from video_audio_sync import ModelManager, process_video  # noqa: E402

app = create_worker_app("video_audio_sync")

_model_mgr = None
_DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _get_model_mgr(device: str = "cpu") -> ModelManager:
    global _model_mgr
    if _model_mgr is None:
        _model_mgr = ModelManager(device=device)
    return _model_mgr


@worker_process_init.connect
def _preload_and_warmup(**kwargs):
    """Preload weights and warmup one stage at a time; park models on CPU idle.

    Keeps startup / idle VRAM low so GPU can be shared with video_embedding.
    """
    global _model_mgr
    t0 = time.time()
    logger.info("[video_audio_sync] preloading models on device={}", _DEFAULT_DEVICE)

    _model_mgr = ModelManager(device=_DEFAULT_DEVICE)

    # Warmup ImageBind, then release so later stages do not share GPU with it
    _model_mgr.load_imagebind()
    device = _model_mgr.device
    dummy_input = {
        "vision": torch.randn(1, 3, 2, 224, 224, device=device),
        "audio": torch.randn(1, 1, 128, 204, device=device),
    }
    with torch.no_grad():
        try:
            _model_mgr.load_imagebind()(dummy_input)
        except Exception:
            pass  # shape mismatch is fine, kernel compilation still happens
    _model_mgr.release_imagebind()

    # Prefetch Synchformer / SyncNet onto CPU (load then immediately release)
    try:
        _model_mgr.load_synchformer()
        _model_mgr.release_synchformer()
    except Exception as e:
        logger.warning("[video_audio_sync] synchformer preload skipped: {}", e)
    try:
        _model_mgr.load_syncnet()
        _model_mgr.release_syncnet()
    except Exception as e:
        logger.warning("[video_audio_sync] syncnet preload skipped: {}", e)

    logger.info("[video_audio_sync] preload + warmup done in {:.2f}s (models parked on CPU)", time.time() - t0)


def run_fn(local_input_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    device = params.pop("device", _DEFAULT_DEVICE)
    params.pop("job_id", None)
    params.pop("object_key", None)
    mgr = _get_model_mgr(device)
    result = process_video(local_input_path, mgr, **params)
    result["algo_name"] = "video_audio_sync"
    result["algo_version"] = "v1_0"
    return result


@app.task(name="algo.video_audio_sync", bind=True, max_retries=2, default_retry_delay=10, acks_late=True)
def run_task(self, task_id: str) -> Dict[str, Any]:
    return execute_algo_task(task_id, celery_task_id=self.request.id, run_fn=run_fn)
