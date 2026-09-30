#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Audio Sync Analysis Pipeline v1.0
Pipeline: ImageBind (semantic alignment) -> Synchformer (temporal sync)
          -> SyncNet (frame-level lip sync)

Usage:
    python video_audio_sync.py --input-json test/test_local_1.json --output-json test/result_local_1.json
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import requests
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.interpolate
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
SYNCHFORMER_DIR = BASE_DIR / "Synchformer"
IMAGEBIND_DIR = BASE_DIR / "ImageBind"
SYNCNET_DIR = BASE_DIR / "syncnet_python"

sys.path.insert(0, str(SYNCHFORMER_DIR))
sys.path.insert(0, str(SYNCNET_DIR))
sys.path.insert(0, str(SYNCHFORMER_DIR / "model" / "modules" / "feat_extractors" / "visual"))
sys.path.insert(0, str(IMAGEBIND_DIR))
_ALGOS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ALGOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGOS_ROOT))
from common.video_resize import DEFAULT_ANALYSIS_MAX_SIDE, maybe_downscale_video  # noqa: E402

# Lazy imports for heavy / optional dependencies
_lazy_imagebind_model = None
_lazy_ModalityType = None
_lazy_imagebind_data = None


def _ensure_imagebind():
    global _lazy_imagebind_model, _lazy_ModalityType, _lazy_imagebind_data
    if _lazy_imagebind_model is None:
        from imagebind.models import imagebind_model as _im
        from imagebind.models.imagebind_model import ModalityType as _mt
        from imagebind import data as _id
        _lazy_imagebind_model = _im
        _lazy_ModalityType = _mt
        _lazy_imagebind_data = _id


_lazy_SyncNetInstance = None


def _ensure_syncnet():
    global _lazy_SyncNetInstance
    if _lazy_SyncNetInstance is None:
        from SyncNetInstance import SyncNetInstance as _sni
        _lazy_SyncNetInstance = _sni


_lazy_synchformer_modules = {}


def _ensure_synchformer():
    if not _lazy_synchformer_modules:
        try:
            from omegaconf import OmegaConf as _oc
            from dataset.dataset_utils import get_video_and_audio as _gvaa
            from dataset.transforms import make_class_grid as _mcg
            from scripts.train_utils import get_model as _gm
            from scripts.train_utils import get_transforms as _gt
            from scripts.train_utils import prepare_inputs as _pi
            _lazy_synchformer_modules.update({
                "OmegaConf": _oc,
                "get_video_and_audio": _gvaa,
                "make_class_grid": _mcg,
                "get_model": _gm,
                "get_transforms": _gt,
                "prepare_inputs": _pi,
            })
        except Exception as exc:
            logger.error("Failed to import Synchformer dependencies: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGEBIND_VIDEO_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGEBIND_VIDEO_STD = (0.26862954, 0.26130258, 0.27577711)
SYNCNET_FPS = 25
SYNCNET_AUDIO_RATE = 16000

SYNCNET_WEIGHT = CHECKPOINTS_DIR / "syncnet_v2.model"
IMAGEBIND_WEIGHT = CHECKPOINTS_DIR / "imagebind_huge.pth"

SYNCHFORMER_CFG_240104 = (
    CHECKPOINTS_DIR / "synchformer" / "24-01-04T16-39-21" / "cfg-24-01-04T16-39-21.yaml"
)
SYNCHFORMER_CKPT_240104 = (
    CHECKPOINTS_DIR / "synchformer" / "24-01-04T16-39-21" / "24-01-04T16-39-21.pt"
)
SYNCHFORMER_CFG_240122 = (
    CHECKPOINTS_DIR / "synchformer" / "24-01-22T20-34-52" / "cfg-24-01-22T20-34-52.yaml"
)
SYNCHFORMER_CKPT_240122 = (
    CHECKPOINTS_DIR / "synchformer" / "24-01-22T20-34-52" / "24-01-22T20-34-52.pt"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _download_video(url: str, timeout: int = 300) -> str:
    """Download video from URL with headers and validation."""
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Try requests first (handles headers / redirects better)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        logger.warning("requests download failed (%s), falling back to urllib", exc)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(local_path, "wb") as f:
                    f.write(resp.read())
        except Exception as exc2:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Failed to download video from {url}: {exc2}")

    # Validate downloaded file
    size = os.path.getsize(local_path)
    if size == 0:
        os.remove(local_path)
        raise RuntimeError(f"Downloaded video is empty: {url}")
    logger.info("Downloaded %d bytes to %s", size, local_path)

    # Sanity check via ffprobe
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            subprocess.run([
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", local_path,
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            os.remove(local_path)
            raise RuntimeError(f"Downloaded file is not a valid video: {url}")
    return local_path


def _reencode_for_model(video_path: str, output_dir: str, vfps: int = 25, afps: int = 16000, in_size: int = 256) -> str:
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    vf = (
        f"fps={vfps},scale=iw*{in_size}/min(iw\\,ih):ih*{in_size}/min(iw\\,ih),"
        "crop=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-vf", vf, "-ar", str(afps), str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)


def _maybe_download_tmp(video_path: str) -> Tuple[str, bool]:
    if _is_url(video_path):
        logger.info("Downloading video from URL: %s", video_path)
        local = _download_video(video_path)
        return local, True
    if not Path(video_path).exists():
        alt = BASE_DIR / video_path
        if alt.exists():
            return str(alt.resolve()), False
        raise FileNotFoundError(f"Video not found: {video_path}")
    return str(Path(video_path).resolve()), False


def _has_audio_stream(video_path: str) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1", video_path,
                ],
                capture_output=True, text=True, check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    try:
        _, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
        audio_fps = info.get("audio_fps", 0)
        return audio_fps is not None and audio_fps > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------
def _release_cuda_cache() -> None:
    """Free cached CUDA blocks between pipeline stages to reduce peak VRAM."""
    import gc

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------
class ModelManager:
    """Owns the three sync models and keeps at most one stage resident on GPU.

    Weights are parked on CPU between stages so peak VRAM ≈ heaviest single
    stage (ImageBind), which is safer when sharing a card with video_embedding.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._imagebind_model = None
        self._synchformer_model = None
        self._synchformer_cfg = None
        self._synchformer_grid = None
        self._syncnet_model = None
        self._synchformer_error = None
        self._syncnet_error = None

    def load_imagebind(self):
        if self._imagebind_model is not None:
            self._imagebind_model.to(self.device)
            return self._imagebind_model
        if not IMAGEBIND_WEIGHT.exists():
            raise FileNotFoundError(f"ImageBind weight not found: {IMAGEBIND_WEIGHT}")
        _ensure_imagebind()
        model = _lazy_imagebind_model.imagebind_huge(pretrained=False)
        state_dict = torch.load(str(IMAGEBIND_WEIGHT), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        self._imagebind_model = model
        logger.info("ImageBind loaded on %s", self.device)
        return model

    def release_imagebind(self):
        if self._imagebind_model is not None:
            self._imagebind_model.to("cpu")
            logger.info("ImageBind released to CPU")
        _release_cuda_cache()

    def _patch_synchformer_cfg(self, cfg):
        cfg.model.params.afeat_extractor.params.ckpt_path = None
        cfg.model.params.vfeat_extractor.params.ckpt_path = None
        cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
            ".modules.feature_selector.", ".sync_model."
        )
        return cfg

    def load_synchformer(self, exp_name: str = "24-01-04T16-39-21"):
        if self._synchformer_model is not None:
            self._synchformer_model.to(self.device)
            return self._synchformer_model, self._synchformer_cfg, self._synchformer_grid
        if self._synchformer_error is not None:
            raise RuntimeError(f"Synchformer previously failed to load: {self._synchformer_error}")

        _ensure_synchformer()
        oc = _lazy_synchformer_modules["OmegaConf"]
        get_model_fn = _lazy_synchformer_modules["get_model"]
        make_grid_fn = _lazy_synchformer_modules["make_class_grid"]

        if exp_name == "24-01-04T16-39-21":
            cfg_path = SYNCHFORMER_CFG_240104
            ckpt_path = SYNCHFORMER_CKPT_240104
        elif exp_name == "24-01-22T20-34-52":
            cfg_path = SYNCHFORMER_CFG_240122
            ckpt_path = SYNCHFORMER_CKPT_240122
        else:
            cfg_path = CHECKPOINTS_DIR / "synchformer" / exp_name / f"cfg-{exp_name}.yaml"
            ckpt_path = CHECKPOINTS_DIR / "synchformer" / exp_name / f"{exp_name}.pt"

        if not cfg_path.exists():
            raise FileNotFoundError(f"Synchformer config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Synchformer checkpoint not found: {ckpt_path}")

        cfg = oc.load(str(cfg_path))
        cfg = self._patch_synchformer_cfg(cfg)
        cfg.ckpt_path = str(ckpt_path)

        _, model = get_model_fn(cfg, self.device)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        max_off_sec = cfg.data.max_off_sec
        num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = make_grid_fn(-max_off_sec, max_off_sec, num_cls)

        self._synchformer_model = model
        self._synchformer_cfg = cfg
        self._synchformer_grid = grid
        logger.info("Synchformer (%s) loaded on %s", exp_name, self.device)
        return model, cfg, grid

    def release_synchformer(self):
        if self._synchformer_model is not None:
            self._synchformer_model.to("cpu")
            logger.info("Synchformer released to CPU")
        _release_cuda_cache()

    def load_syncnet(self):
        if self._syncnet_model is not None:
            self._syncnet_model.to(self.device)
            self._syncnet_model.device = str(self.device)
            return self._syncnet_model
        if self._syncnet_error is not None:
            raise RuntimeError(f"SyncNet previously failed to load: {self._syncnet_error}")

        _ensure_syncnet()
        if not SYNCNET_WEIGHT.exists():
            raise FileNotFoundError(f"SyncNet weight not found: {SYNCNET_WEIGHT}")
        model = _lazy_SyncNetInstance(device=str(self.device))
        model.loadParameters(str(SYNCNET_WEIGHT))
        model.eval()
        self._syncnet_model = model
        logger.info("SyncNet loaded on %s", self.device)
        return model

    def release_syncnet(self):
        if self._syncnet_model is not None:
            self._syncnet_model.to("cpu")
            self._syncnet_model.device = "cpu"
            logger.info("SyncNet released to CPU")
        _release_cuda_cache()


# ---------------------------------------------------------------------------
# ImageBind semantic alignment
# ---------------------------------------------------------------------------
def _extract_imagebind_features(video_path: str, model_mgr: ModelManager, sample_fps: float = 2.0):
    model = model_mgr.load_imagebind()
    device = model_mgr.device

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if orig_fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    frame_interval = max(1, int(orig_fps / sample_fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        idx += 1
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGEBIND_VIDEO_MEAN, std=IMAGEBIND_VIDEO_STD),
    ])
    video_frames = [image_transform(Image.fromarray(f)) for f in frames]

    # Prepare audio via temporary wav
    tmp_wav = None
    audio_emb = None
    try:
        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        ffmpeg = shutil.which("ffmpeg")
        try:
            subprocess.run([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le", tmp_wav,
            ], capture_output=True, text=True, check=True)
            audio_features = _lazy_imagebind_data.load_and_transform_audio_data(
                [tmp_wav],
                device=str(device),
                num_mel_bins=128,
                target_length=204,
                sample_rate=16000,
                clip_duration=2,
                clips_per_video=3,
            )
            with torch.no_grad():
                audio_emb = model({_lazy_ModalityType.AUDIO: audio_features})[_lazy_ModalityType.AUDIO]
                audio_emb = audio_emb.mean(dim=0, keepdim=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            if "does not contain any stream" in stderr:
                logger.warning("No audio stream found in %s, using default audio embedding", video_path)
                audio_emb = None
            else:
                raise
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    # Smaller batch keeps ImageBind activation peak within shared-GPU budgets (~16GB).
    VISION_BATCH = 4
    emb_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(video_frames), VISION_BATCH):
            batch = torch.stack(video_frames[i : i + VISION_BATCH]).to(device)
            chunk_emb = model({_lazy_ModalityType.VISION: batch})[_lazy_ModalityType.VISION]
            emb_chunks.append(chunk_emb.cpu())
            del batch, chunk_emb

    video_emb = torch.cat(emb_chunks, dim=0).mean(dim=0, keepdim=True)
    if audio_emb is not None:
        cos_sim = F.cosine_similarity(video_emb, audio_emb.cpu(), dim=-1).item()
    else:
        cos_sim = 0.0
    return video_emb, audio_emb, cos_sim


# ---------------------------------------------------------------------------
# Synchformer temporal sync
# ---------------------------------------------------------------------------
def _get_video_duration(video_path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            pass


def _truncate_video(video_path, output_dir, max_duration):
    """Truncate video+audio to max_duration seconds (from 0) via ffmpeg."""
    stem = Path(video_path).stem
    out_path = Path(output_dir) / f"{stem}_truncated_{max_duration}s.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-t", str(max_duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    cmd_copy = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-t", str(max_duration),
        "-c", "copy",
        str(out_path),
    ]
    ret2 = subprocess.run(cmd_copy, capture_output=True)
    if ret2.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
        return str(out_path)

    raise RuntimeError(
        f"Failed to truncate video (ffmpeg error): {ret.stderr.decode(errors='ignore').strip()}"
    )


def _synchformer_infer(video_path: str, model_mgr: ModelManager, exp_name: str = "24-01-04T16-39-21", v_start: float = 0.0):
    model, cfg, grid = model_mgr.load_synchformer(exp_name)
    device = model_mgr.device

    # Check video duration (Synchformer requires at least 5s)
    duration = _get_video_duration(video_path)
    if duration > 0 and duration < 5.0:
        raise ValueError(f"Video too short for Synchformer: {duration:.2f}s < 5.0s")

    # Check if re-encoding needed
    try:
        v, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
        _, H, W, _ = v.shape
        vfps = info.get("video_fps", 25)
        afps = info.get("audio_fps", 16000)
    except Exception:
        vfps, afps, H, W = 0, 0, 0, 0

    needs_reencode = vfps != 25 or afps != 16000 or min(H, W) != 256
    reencoded_path = video_path
    tmp_reenc = None
    tmp_dir_obj = None
    if needs_reencode:
        tmp_dir_obj = tempfile.mkdtemp()
        reencoded_path = _reencode_for_model(video_path, tmp_dir_obj)
        tmp_reenc = reencoded_path

    try:
        get_video_and_audio = _lazy_synchformer_modules["get_video_and_audio"]
        get_transforms = _lazy_synchformer_modules["get_transforms"]
        prepare_inputs = _lazy_synchformer_modules["prepare_inputs"]

        rgb, audio, meta = get_video_and_audio(reencoded_path, get_meta=True)
        item = dict(
            video=rgb, audio=audio, meta=meta, path=reencoded_path, split="test",
            targets={"v_start_i_sec": v_start, "offset_sec": 0.0},
        )
        transforms = get_transforms(cfg, which_transforms=["test"])
        item = transforms["test"](item)
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, device)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=cfg.training.use_half_precision):
                _, logits = model(vid, aud)

        logits = logits[0]
        off_probs = F.softmax(logits, dim=-1)
        best_idx = int(torch.argmax(off_probs).item())
        best_offset = float(grid[best_idx])
        best_confidence = float(off_probs[best_idx].item())

        half_sec_range = max(1, int(0.5 / abs(grid[1] - grid[0])))
        center_idx = len(grid) // 2
        low_idx = max(0, center_idx - half_sec_range)
        high_idx = min(len(grid), center_idx + half_sec_range + 1)
        in_sync_conf = float(off_probs[low_idx:high_idx].sum().item())
    finally:
        if tmp_reenc and os.path.exists(tmp_reenc):
            os.remove(tmp_reenc)

    return {
        "offset_sec": best_offset,
        "confidence": best_confidence,
        "in_sync_confidence": in_sync_conf,
        "exp_name": exp_name,
    }


# ---------------------------------------------------------------------------
# SyncNet frame-level sync
# ---------------------------------------------------------------------------
def _syncnet_infer(video_path: str, model_mgr: ModelManager, lip_sync_threshold: float = 5.0):
    model = model_mgr.load_syncnet()
    device = model_mgr.device
    ffmpeg = shutil.which("ffmpeg")

    work_dir = tempfile.mkdtemp(prefix="syncnet_work_")
    video_stem = Path(video_path).stem
    ref = video_stem

    tmp_dir = os.path.join(work_dir, "pytmp", ref)
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract frames at 25fps
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-r", str(SYNCNET_FPS), "-q:v", "2", "-f", "image2",
        os.path.join(tmp_dir, "%06d.jpg"),
    ], check=True)

    # Extract audio to 16kHz mono wav
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-async", "1", "-ac", "1", "-vn",
        "-acodec", "pcm_s16le", "-ar", str(SYNCNET_AUDIO_RATE),
        os.path.join(tmp_dir, "audio.wav"),
    ], check=True)

    # Face Detection (S3FD)
    from detectors import S3FD
    flist = sorted(Path(tmp_dir).glob("*.jpg"))
    if len(flist) == 0:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise ValueError("No frames extracted for SyncNet processing.")

    face_detector = S3FD(device=device)
    face_boxes = []
    for fidx, fname in enumerate(flist):
        img = cv2.imread(str(fname))
        if img is None:
            face_boxes.append([])
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = face_detector.detect_faces(img_rgb, conf_th=0.9, scales=[0.25])
        face_boxes.append([
            {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": float(bbox[-1])}
            for bbox in bboxes
        ])

    # Track faces
    def bb_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    tracks = []
    tmp_faces = [list(f) for f in face_boxes]
    iouThres = 0.5
    min_track = max(50, int(len(flist) * 0.1))
    while True:
        track = []
        for frame_faces in tmp_faces:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= 25:
                    if bb_iou(face["bbox"], track[-1]["bbox"]) > iouThres:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > min_track:
            bboxes_arr = np.array([f["bbox"] for f in track])
            framenum = np.array([f["frame"] for f in track])
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            if len(frame_i) >= min_track:
                bboxes_i = []
                for ij in range(4):
                    interpfn = scipy.interpolate.interp1d(
                        framenum, bboxes_arr[:, ij], kind="linear", fill_value="extrapolate"
                    )
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    if not tracks:
        all_faces = []
        for ff in face_boxes:
            all_faces.extend(ff)
        if not all_faces:
            shutil.rmtree(work_dir, ignore_errors=True)
            return {
                "error": "No faces detected in video",
                "av_offset_frames": None,
                "min_distance": None,
                "confidence": None,
                "is_lip_sync": False,
            }
        med_box = np.median([f["bbox"] for f in all_faces[:min(100, len(all_faces))]], axis=0)
        frame_i = np.arange(len(flist))
        bboxes_i = np.tile(med_box, (len(frame_i), 1))
        tracks.append({"frame": frame_i, "bbox": bboxes_i})

    best_track = max(tracks, key=lambda t: len(t["frame"]))

    # Crop lip region and write avi
    dets = {"x": [], "y": [], "s": []}
    for det in best_track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    for key in dets:
        if len(dets[key]) >= 13:
            dets[key] = scipy.signal.medfilt(dets[key], kernel_size=13)

    crop_dir = os.path.join(work_dir, "crop_tmp")
    os.makedirs(crop_dir, exist_ok=True)
    crop_avi = os.path.join(crop_dir, f"{ref}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(crop_avi, fourcc, SYNCNET_FPS, (224, 224))

    for fidx, frame_idx in enumerate(best_track["frame"]):
        if int(frame_idx) >= len(flist):
            continue
        img = cv2.imread(str(flist[int(frame_idx)]))
        if img is None:
            continue
        cs = 0.4
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        my = dets["y"][fidx]
        mx = dets["x"][fidx]
        padded = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my += bsi
        mx += bsi
        y1 = max(0, int(my - bs * (1 + cs)))
        y2 = min(padded.shape[0], int(my + bs * (1 + cs)))
        x1 = max(0, int(mx - bs * (1 + cs)))
        x2 = min(padded.shape[1], int(mx + bs * (1 + cs)))
        crop = padded[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224, 224))
        vw.write(crop)
    vw.release()

    # Merge audio into cropped avi
    merged_avi = os.path.join(crop_dir, f"{ref}_merged.avi")
    audio_wav_src = os.path.join(tmp_dir, "audio.wav")
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", crop_avi, "-i", audio_wav_src,
        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-shortest", merged_avi,
    ], check=True)

    # Run SyncNet evaluate
    opt_eval = SimpleNamespace(
        tmp_dir=os.path.join(work_dir, "pytmp"),
        reference=ref,
        batch_size=20,
        vshift=15,
    )
    offset, confidence, dists = model.evaluate(opt_eval, videofile=merged_avi)

    av_offset_frames = int(offset.item()) if hasattr(offset, "item") else int(offset)
    confidence_val = float(confidence.item()) if hasattr(confidence, "item") else float(confidence)
    min_distance = float(dists.min()) if hasattr(dists, "min") else float(np.min(dists))

    shutil.rmtree(work_dir, ignore_errors=True)

    is_lip_sync = confidence_val >= lip_sync_threshold and abs(av_offset_frames) <= 1
    return {
        "av_offset_frames": av_offset_frames,
        "min_distance": round(min_distance, 4),
        "confidence": round(confidence_val, 4),
        "is_lip_sync": bool(is_lip_sync),
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model_mgr: ModelManager,
    sample_fps: float = 2.0,
    enable_semantic: bool = True,
    enable_temporal: bool = True,
    enable_frame_level: bool = True,
    semantic_threshold: float = 0.2,
    temporal_offset_threshold: float = 0.5,
    frame_level_threshold: float = 5.0,
    sync_threshold: float = 0.5,
    min_duration: float = 5.0,
    max_duration: float = 60.0,
    max_side: Optional[int] = DEFAULT_ANALYSIS_MAX_SIDE,
) -> Dict[str, Any]:
    timings: Dict[str, Optional[float]] = {}
    t0_overall = time.time()

    local_path, is_temp = _maybe_download_tmp(video_path)
    if is_temp:
        timings["download_video"] = 0.0
        t_dl = time.time()
    else:
        t_dl = None

    has_audio = _has_audio_stream(local_path)
    duration = _get_video_duration(local_path)
    result: Dict[str, Any] = {"video_path": video_path, "has_audio": bool(has_audio)}

    # Truncate if video is longer than max_duration
    truncated_path = None
    truncate_tmp_dir = None
    if duration > max_duration:
        logger.info("Video duration %.2fs exceeds max_duration %.2fs, truncating", duration, max_duration)
        truncate_tmp_dir = tempfile.mkdtemp(prefix="video_trunc_")
        try:
            truncated_path = _truncate_video(local_path, truncate_tmp_dir, max_duration)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            logger.warning("Truncation failed (%s), using original video", exc)
            truncated_path = local_path
    else:
        truncated_path = local_path

    if not has_audio:
        logger.info("No audio stream detected in %s, returning defaults", video_path)
        if is_temp and os.path.exists(local_path):
            os.remove(local_path)
        if t_dl is not None:
            timings["download_video"] = round(time.time() - t_dl, 3)
        null_semantic = {"cosine_similarity": 0.0, "is_semantic_sync": False} if enable_semantic else None
        null_temporal = {"offset_seconds": None, "confidence": None, "is_temporal_sync": False} if enable_temporal else None
        null_frame = {"av_offset_frames": None, "min_distance": None, "confidence": None, "is_lip_sync": False} if enable_frame_level else None
        result.update({
            "semantic_alignment": null_semantic,
            "temporal_sync": null_temporal,
            "frame_level_sync": null_frame,
            "overall_sync_score": 0.0,
            "overall_is_sync": False,
            "timings": timings,
            "algo_name": "video_audio_sync",
            "algo_version": "v1_0",
        })
        return result

    downscale_tmp_dir = None
    analysis_path = truncated_path
    try:
        analysis_path, downscale_tmp_dir = maybe_downscale_video(truncated_path, max_side)
    except Exception as exc:
        logger.warning("max_side downscale failed (%s), using original resolution", exc)
        analysis_path = truncated_path
        downscale_tmp_dir = None

    # Stage-serial GPU residency: only one model family on device at a time.
    # ---- Semantic Alignment (ImageBind) ----
    if enable_semantic:
        model_mgr.release_synchformer()
        model_mgr.release_syncnet()
        t1 = time.time()
        try:
            _, _, cos_sim = _extract_imagebind_features(analysis_path, model_mgr, sample_fps=sample_fps)
        finally:
            model_mgr.release_imagebind()
        t_elapsed = time.time() - t1
        timings["semantic_alignment"] = round(t_elapsed, 3)
        timings["extract_video_features"] = round(t_elapsed * 0.55, 3)
        timings["extract_audio_features"] = round(t_elapsed * 0.35, 3)
        is_semantic_sync = cos_sim > semantic_threshold
        result["semantic_alignment"] = {
            "cosine_similarity": round(cos_sim, 4),
            "is_semantic_sync": bool(is_semantic_sync),
        }
    else:
        timings["semantic_alignment"] = None
        timings["extract_video_features"] = None
        timings["extract_audio_features"] = None
        result["semantic_alignment"] = None

    # ---- Temporal Sync (Synchformer) ----
    if enable_temporal:
        model_mgr.release_imagebind()
        model_mgr.release_syncnet()
        t2 = time.time()
        local_synch = local_path
        tmp_reenc = None
        tmp_dir_obj = None
        try:
            v, _, info = torchvision.io.read_video(analysis_path, pts_unit="sec")
            _, H, W, _ = v.shape
            vfps = info.get("video_fps", 25)
            afps = info.get("audio_fps", 16000)
            if vfps != 25 or afps != 16000 or min(H, W) != 256:
                tmp_dir_obj = tempfile.mkdtemp()
                local_synch = _reencode_for_model(analysis_path, tmp_dir_obj)
                tmp_reenc = local_synch
        except Exception:
            local_synch = analysis_path

        try:
            temporal_res = _synchformer_infer(local_synch, model_mgr, exp_name="24-01-04T16-39-21")
        except Exception as exc:
            logger.warning("Synchformer inference failed: %s", exc)
            temporal_res = None
        finally:
            if tmp_reenc and os.path.exists(tmp_reenc):
                os.remove(tmp_reenc)
            model_mgr.release_synchformer()

        if temporal_res is not None:
            timings["temporal_sync"] = round(time.time() - t2, 3)
            offset_seconds = round(temporal_res["offset_sec"], 4)
            is_temporal_sync = abs(offset_seconds) < temporal_offset_threshold
            result["temporal_sync"] = {
                "offset_seconds": offset_seconds,
                "confidence": round(temporal_res["confidence"], 4),
                "is_temporal_sync": bool(is_temporal_sync),
            }
        else:
            timings["temporal_sync"] = None
            result["temporal_sync"] = {"error": "Synchformer inference failed", "offset_seconds": None,
                                        "confidence": None, "is_temporal_sync": False}
    else:
        timings["temporal_sync"] = None
        result["temporal_sync"] = None

    # ---- Frame-level Sync (SyncNet) ----
    if enable_frame_level:
        model_mgr.release_imagebind()
        model_mgr.release_synchformer()
        t3 = time.time()
        try:
            frame_res = _syncnet_infer(analysis_path, model_mgr, lip_sync_threshold=frame_level_threshold)
        except Exception as exc:
            logger.warning("SyncNet inference failed: %s", exc)
            frame_res = {
                "error": str(exc),
                "av_offset_frames": None,
                "min_distance": None,
                "confidence": None,
                "is_lip_sync": False,
            }
        finally:
            model_mgr.release_syncnet()
        timings["frame_level_sync"] = round(time.time() - t3, 3)
        result["frame_level_sync"] = {
            "av_offset_frames": frame_res.get("av_offset_frames"),
            "min_distance": frame_res.get("min_distance"),
            "confidence": frame_res.get("confidence"),
            "is_lip_sync": frame_res.get("is_lip_sync", False),
        }
        if "error" in frame_res:
            result["frame_level_sync"]["error"] = frame_res["error"]
    else:
        timings["frame_level_sync"] = None
        result["frame_level_sync"] = None

    # Cleanup
    if is_temp and os.path.exists(local_path):
        os.remove(local_path)
    if truncate_tmp_dir and os.path.exists(truncate_tmp_dir):
        shutil.rmtree(truncate_tmp_dir, ignore_errors=True)
    if downscale_tmp_dir and os.path.exists(downscale_tmp_dir):
        shutil.rmtree(downscale_tmp_dir, ignore_errors=True)
    if t_dl is not None:
        timings["download_video"] = round(time.time() - t_dl, 3)

    # ---- Overall Score ----
    scores = []
    weights = []
    if enable_semantic and result["semantic_alignment"] is not None:
        scores.append(max(0.0, result["semantic_alignment"]["cosine_similarity"]))
        weights.append(0.3)
    if enable_temporal and result["temporal_sync"] is not None and result["temporal_sync"].get("offset_seconds") is not None:
        off_score = max(0.0, 1.0 - abs(result["temporal_sync"]["offset_seconds"]) / 2.0)
        scores.append(off_score * result["temporal_sync"]["confidence"])
        weights.append(0.35)
    if enable_frame_level and result["frame_level_sync"] is not None:
        fl = result["frame_level_sync"]
        if fl.get("confidence") is not None:
            lip_score = min(1.0, fl["confidence"] / 10.0)
            scores.append(lip_score)
            weights.append(0.35)

    if scores:
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        overall_score = 0.0

    overall_is_sync = overall_score >= sync_threshold

    timings["load_models"] = None
    timings["total"] = round(time.time() - t0_overall, 3)

    result["overall_sync_score"] = round(overall_score, 4)
    result["overall_is_sync"] = bool(overall_is_sync)
    result["timings"] = timings
    result["algo_name"] = "video_audio_sync"
    result["algo_version"] = "v1_0"

    _release_cuda_cache()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Audio Sync Analysis v1.0")
    parser.add_argument("--input-json", required=True, help="Path to input JSON file")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for inference")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Video sampling FPS for ImageBind")
    parser.add_argument("--enable-semantic", type=lambda x: x.lower() == "true", default=True, help="Enable ImageBind")
    parser.add_argument("--enable-temporal", type=lambda x: x.lower() == "true", default=True, help="Enable Synchformer")
    parser.add_argument("--enable-frame-level", type=lambda x: x.lower() == "true", default=True, help="Enable SyncNet")
    parser.add_argument("--semantic-threshold", type=float, default=0.2, help="ImageBind cos-sim threshold")
    parser.add_argument("--temporal-offset-threshold", type=float, default=0.5, help="Synchformer offset threshold (sec)")
    parser.add_argument("--frame-level-threshold", type=float, default=5.0, help="SyncNet confidence threshold")
    parser.add_argument("--sync-threshold", type=float, default=0.5, help="Overall sync score threshold")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum video duration in seconds for Synchformer (default: 5)")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Maximum video duration in seconds; longer videos will be truncated (default: 60)")
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_SIDE,
        help="Max longest side for analysis video (aspect-preserving; 0 disables)",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    video_path = inp.get("video_path")
    if not video_path:
        raise ValueError("input JSON must contain 'video_path'")

    print(f"[INFO] Processing video: {video_path}")

    if torch.cuda.is_available() and args.device == "cuda":
        print("Using GPU for inference.")
        device = "cuda"
    else:
        print("Using CPU for inference.")
        device = "cpu"

    t_load = time.time()
    model_mgr = ModelManager(device=device)
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
    load_time = time.time() - t_load

    result = process_video(
        video_path=video_path,
        model_mgr=model_mgr,
        sample_fps=args.sample_fps,
        enable_semantic=args.enable_semantic,
        enable_temporal=args.enable_temporal,
        enable_frame_level=args.enable_frame_level,
        semantic_threshold=args.semantic_threshold,
        temporal_offset_threshold=args.temporal_offset_threshold,
        frame_level_threshold=args.frame_level_threshold,
        sync_threshold=args.sync_threshold,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_side=args.max_side,
    )
    result["timings"]["load_models"] = round(load_time, 3)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Result saved to: {args.output_json}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
