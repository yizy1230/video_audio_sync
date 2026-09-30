# Overview

Decide whether the audio and the picture in a video are in sync.

# v1_0

The pipeline follows the data-processing choices used by LTX-2.3 and MOVA: Synchformer, ImageBind, and SyncNet.

ImageBind maps video and audio into one embedding space. Synchformer models temporal alignment. SyncNet applies a frame-level constraint. The three stages go from coarse to fine.

## Submodules

**ImageBind** maps video and audio into the same embedding space and scores semantic similarity. See https://github.com/facebookresearch/ImageBind.

**Synchformer** predicts an audio-visual time offset and a confidence from sparse clip features. See https://github.com/v-iashin/Synchformer.

**syncnet_python** measures frame-level lip sync and confidence on a face video. See https://github.com/joonson/syncnet_python.

## Checkpoints

- `24-01-04T16-39-21`: offset and score on real video
- `24-01-22T20-34-52`: binary in-sync / out-of-sync score on generated video

# Development

## Local environment

```bash
conda create -n video_audio_sync python=3.8.16 -y
conda activate video_audio_sync
conda install -c conda-forge av==9.0.0
pip install -r requirements.txt
# web UI
pip install flask fastapi loguru gradio
# profiling
pip install psutil memory_profiler
# Jupyter
pip install ipykernel && python -m ipykernel install --user --name video_audio_sync --display-name "video_audio_sync"
```

## Local scripts

`video_audio_sync.py`

Input JSON. `video_path` may be a local file or a URL.

```json
{
  "video_path": "xxxx"
}
```

Output JSON:

```json
{
  "video_path": "xxxx",
  "has_audio": true,
  "semantic_alignment": {
    "cosine_similarity": 0.82,
    "is_semantic_sync": true
  },
  "temporal_sync": {
    "offset_seconds": 120,
    "confidence": 0.91,
    "is_temporal_sync": true
  },
  "frame_level_sync": {
    "av_offset_frames": 3,
    "min_distance": 5.35,
    "confidence": 10.02,
    "is_lip_sync": true
  },
  "overall_sync_score": 0.88,
  "overall_is_sync": true,
  "timings": {
    "load_models": 2.5,
    "extract_video_features": 1.2,
    "extract_audio_features": 0.8,
    "semantic_alignment": 0.1,
    "temporal_sync": 0.5,
    "frame_level_sync": 3.0,
    "download_video": 2.0,
    "total": 10.1
  }
}
```

```bash
CUDA_VISIBLE_DEVICES=1
python video_audio_sync.py --input-json test/test_local_1.json --output-json test/result_local_1.json
python video_audio_sync.py --input-json test/test_local_2.json --output-json test/result_local_2.json
python video_audio_sync.py --input-json test/test_local_3.json --output-json test/result_local_3.json
python video_audio_sync.py --input-json test/test_url.json --output-json test/result_url.json
```

`video_audio_sync_server.py` starts a FastAPI server.

Routes:

- `GET /ready` — service is ready when the status is 200
- `GET /test` — run the predefined test cases
- `POST /video_audio_sync` — score one video

Request body:

```json
{
  "video_path": "path_or_url_to_video"
}
```

The response has the same fields as the CLI output, plus `algo_name` and `algo_version`. `timings.load_models` is null because the server loads models at startup.

```bash
python video_audio_sync_server.py --device cuda --port 8000
```

```bash
python3 test/test_api.py --input-json test/test_local_2.json --output-dir test/server_results
python3 test/test_api.py --input-json test/test_local_3.json --output-dir test/server_results
python3 test/test_api.py --input-json test/test_url.json --output-dir test/server_results
```

## Parameters

`video_audio_sync.py` (CLI)

- `--input-json` — input JSON path (required)
- `--output-json` — output JSON path (required)
- `--device` — `cpu` or `cuda` (default: `cuda`)
- `--sample-fps` — ImageBind video sample rate (default: 4.0). It affects GPU memory and speed.
- `--enable-semantic` — run ImageBind semantic alignment (default: true)
- `--enable-temporal` — run Synchformer temporal sync (default: true)
- `--enable-frame-level` — run SyncNet frame-level sync (default: true)
- `--semantic-threshold` — ImageBind cosine threshold (default: 0.2)
- `--temporal-offset-threshold` — Synchformer offset threshold, in seconds (default: 0.5)
- `--frame-level-threshold` — SyncNet confidence threshold (default: 5.0)
- `--sync-threshold` — overall score threshold; below it the clip is out of sync (default: 0.5)
- `--min-duration` — minimum duration Synchformer accepts, in seconds (default: 5.0)
- `--max-duration` — maximum processed duration; longer video is truncated, in seconds (default: 60.0)

`video_audio_sync_server.py` (FastAPI) accepts the same analysis flags, plus:

- `--host` — bind address (default: `0.0.0.0`)
- `--port` — bind port (default: 8000)

`POST /video_audio_sync` body

- `video_path` — local path or URL (required)
- `sample_fps`, `enable_semantic`, `enable_temporal`, `enable_frame_level`
- `semantic_threshold`, `temporal_offset_threshold`, `frame_level_threshold`, `sync_threshold`
- `min_duration`, `max_duration`

Omitted fields use the values the server was started with.

## Evaluation

## Profiling

## Demo

```bash
python3 video_audio_sync_webui.py
```

# Service

## Docker

## Request and response

## Start command

# Follow-ups

## Build

Build the worker from this repository root:

```bash
docker build -f Dockerfile -t algo-video_audio_sync:v1_0 .
```

## Weights

Weight files are not stored in this repository. Download URLs and destination paths are in `checkpoints/WEIGHTS.txt`.

## Citation

```bibtex
@software{video_audio_sync,
  title   = {video_audio_sync},
  year    = {2026},
  author  = {yizy},
  license = {Apache-2.0}
}
```

## License

Code written for this repository is under the Apache License 2.0. See [LICENSE](LICENSE). Third-party directories keep their own licenses. See [NOTICE](NOTICE).

## Release

- **v1.0.0** — First public release. Audio-visual sync with ImageBind, Synchformer, and SyncNet, plus a Celery worker.

## Future Work

- Make Synchformer and SyncNet feature extraction swappable so the vendored third-party trees can shrink.
