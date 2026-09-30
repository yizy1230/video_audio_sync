#!/usr/bin/env bash
# =============================================================================
# Download pre-trained weights for video_audio_sync v1.0
# ImageBind + Synchformer + SyncNet + S3FD face detector
# Target dir: ./checkpoints/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

# ---------------------------------------------------------------------------
# 1. ImageBind (imagebind_huge.pth)
# ---------------------------------------------------------------------------
IMAGEBIND_URL="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
IMAGEBIND_DST="${CKPT_DIR}/imagebind_huge.pth"

echo "[1/4] Downloading ImageBind weights ..."
if [ -f "${IMAGEBIND_DST}" ]; then
    echo "    Already exists: ${IMAGEBIND_DST}"
else
    wget -q --show-progress "${IMAGEBIND_URL}" -O "${IMAGEBIND_DST}" || \
        curl -L -o "${IMAGEBIND_DST}" "${IMAGEBIND_URL}"
    echo "    Saved to: ${IMAGEBIND_DST}"
fi

# ---------------------------------------------------------------------------
# 2. SyncNet (syncnet_v2.model)
# ---------------------------------------------------------------------------
SYNCNET_URL="https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
SYNCNET_DST="${CKPT_DIR}/syncnet_v2.model"

echo "[2/4] Downloading SyncNet weights ..."
if [ -f "${SYNCNET_DST}" ]; then
    echo "    Already exists: ${SYNCNET_DST}"
else
    wget -q --show-progress "${SYNCNET_URL}" -O "${SYNCNET_DST}" || \
        curl -L -o "${SYNCNET_DST}" "${SYNCNET_URL}"
    echo "    Saved to: ${SYNCNET_DST}"
fi

# ---------------------------------------------------------------------------
# 3. Synchformer config + checkpoint
# ---------------------------------------------------------------------------
echo "[3/4] Downloading Synchformer weights ..."

SYNCHFORMER_NAME="24-01-04T16-39-21"
SYNCHFORMER_CFG="cfg-${SYNCHFORMER_NAME}.yaml"
SYNCHFORMER_PT="${SYNCHFORMER_NAME}.pt"

mkdir -p "${CKPT_DIR}/synchformer"

PARENT_LINK="https://a3s.fi/swift/v1/AUTH_a235c0f452d6488f830be462b58da78c/sync_models"

if [ -f "${CKPT_DIR}/synchformer/${SYNCHFORMER_CFG}" ]; then
    echo "    Already exists: ${CKPT_DIR}/synchformer/${SYNCHFORMER_CFG}"
else
    wget -q --show-progress "${PARENT_LINK}/${SYNCHFORMER_NAME}/${SYNCHFORMER_CFG}" \
        -O "${CKPT_DIR}/synchformer/${SYNCHFORMER_CFG}" || \
        curl -L -o "${CKPT_DIR}/synchformer/${SYNCHFORMER_CFG}" \
            "${PARENT_LINK}/${SYNCHFORMER_NAME}/${SYNCHFORMER_CFG}"
    echo "    Saved to: ${CKPT_DIR}/synchformer/${SYNCHFORMER_CFG}"
fi

if [ -f "${CKPT_DIR}/synchformer/${SYNCHFORMER_PT}" ]; then
    echo "    Already exists: ${CKPT_DIR}/synchformer/${SYNCHFORMER_PT}"
else
    wget -q --show-progress "${PARENT_LINK}/${SYNCHFORMER_NAME}/${SYNCHFORMER_PT}" \
        -O "${CKPT_DIR}/synchformer/${SYNCHFORMER_PT}" || \
        curl -L -o "${CKPT_DIR}/synchformer/${SYNCHFORMER_PT}" \
            "${PARENT_LINK}/${SYNCHFORMER_NAME}/${SYNCHFORMER_PT}"
    echo "    Saved to: ${CKPT_DIR}/synchformer/${SYNCHFORMER_PT}"
fi

# ---------------------------------------------------------------------------
# 4. SyncNet S3FD face detector weight
# ---------------------------------------------------------------------------
S3FD_URL="https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth"
S3FD_DST="${CKPT_DIR}/sfd_face.pth"

echo "[4/4] Downloading S3FD face detector weights ..."
if [ -f "${S3FD_DST}" ]; then
    echo "    Already exists: ${S3FD_DST}"
else
    wget -q --show-progress "${S3FD_URL}" -O "${S3FD_DST}" || \
        curl -L -o "${S3FD_DST}" "${S3FD_URL}"
    echo "    Saved to: ${S3FD_DST}"
fi

# Symlink so the detector __init__.py can find it
S3FD_WEIGHTS_DIR="${SCRIPT_DIR}/syncnet_python/detectors/s3fd/weights"
mkdir -p "${S3FD_WEIGHTS_DIR}"
if [ ! -f "${S3FD_WEIGHTS_DIR}/sfd_face.pth" ]; then
    ln -s "${S3FD_DST}" "${S3FD_WEIGHTS_DIR}/sfd_face.pth"
    echo "    Symlinked to: ${S3FD_WEIGHTS_DIR}/sfd_face.pth"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "All weights downloaded to: ${CKPT_DIR}"
find "${CKPT_DIR}" -type f
