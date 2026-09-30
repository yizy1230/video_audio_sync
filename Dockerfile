# video_audio_sync worker
# Python 3.10 | GPU (PyTorch 2.1 + CUDA 11.8) — ImageBind + Synchformer + SyncNet
#
# Build (from algos/ root):
#   docker build -f video_audio_sync/v1_0/Dockerfile -t algo-video_audio_sync:v1_0 .
#
# Run:
#   docker run -d --gpus all --name worker-video_audio_sync \
#     -e CELERY_BROKER_URL=amqp://guest:guest@<host>:5672// \
#     -e PG_HOST=<host> -e MINIO_ENDPOINT=<host>:9000 \
#     -v /data/cache:/data/cache/pipeline \
#     algo-video_audio_sync:v1_0
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# China mirrors: apt
# China mirrors: pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# av==9.0.0 via conda-forge (pip source build fails; av>=9.1.1 breaks Synchformer accuracy)
# RUN conda install -c conda-forge av==9.0.0 -y \
#     && conda clean -afy
# RUN conda install -c conda-forge av==8.1.0 -y \
#     && conda clean -afy
RUN conda install -y -c conda-forge av==9.0.0 \
  && conda clean -afy

# Worker framework + critical runtime deps (installed separately so failures are not masked)
RUN pip install --no-cache-dir \
    celery sqlalchemy psycopg2-binary minio pyyaml loguru requests \
    opencv-python==5.0.0.93

# Algo dependencies (torch/torchvision/torchaudio from base image; pytorchvideo installed after COPY;
# av via conda above; opencv-python already installed above)
COPY requirements.txt /tmp/reqs.txt
RUN printf "numpy<2\ntorch==2.1.0\ntorchvision==0.16.0\npyarrow<17\njax==0.4.13\njaxlib==0.4.13\n" > /tmp/constraints.txt \
    && grep -v -E '^(torch==|torchvision==|torchaudio==|--extra-index-url|pytorchvideo|opencv-python|numpy|openai|fastapi|uvicorn|gradio|pydantic|vtk)' /tmp/reqs.txt > /tmp/reqs_filtered.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/reqs_filtered.txt

COPY common /app/common
COPY . /app/algo
WORKDIR /app/algo

# pytorchvideo from vendored source (local file:// path in requirements.txt doesn't work in Docker)
RUN pip install --no-cache-dir --no-deps ./pytorchvideo-28fe037d212663c6a24f373b94cc5d478c8c1a1d || true

ENV PYTHONPATH=/app:/app/algo
ENV WORKER_CONFIG_PATH=/app/algo/worker_config.yaml
ENV DEVICE=cuda

CMD ["celery", "-A", "video_audio_sync_worker.app", "worker", \
     "--pool=solo", "-Q", "video_audio_sync", "-c", "1", "-n", "video_audio_sync@%h"]
