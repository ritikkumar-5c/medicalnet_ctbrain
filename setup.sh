#!/usr/bin/env bash
# Set up the training environment for the MedicalNet CT-brain classifier.
# Intended for a GCP NVIDIA/CUDA GPU instance (Linux, CUDA 12.x).
#
# Usage:
#   bash setup.sh            # CUDA 12.1 wheels (default)
#   CUDA=cpu bash setup.sh   # CPU-only torch (for local wiring/debug)
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJ_DIR/.venv}"
CUDA="${CUDA:-cu121}"

echo ">> Creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

if [ "$CUDA" = "cpu" ]; then
  echo ">> Installing CPU torch"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  echo ">> Installing CUDA ($CUDA) torch"
  pip install torch --index-url "https://download.pytorch.org/whl/$CUDA"
fi

echo ">> Installing project requirements"
pip install -r "$PROJ_DIR/requirements.txt"

# Optional extra DICOM decoder fallback. pylibjpeg (in requirements) already
# covers JPEG/JPEG-LS/JPEG2000; gdcm is a best-effort bonus and may have no
# wheel on some platforms, so don't fail the whole setup if it can't build.
echo ">> Installing python-gdcm (optional, best-effort)"
pip install python-gdcm || echo "   (python-gdcm unavailable here — pylibjpeg already covers decoding)"

echo ">> Sanity check"
python -c "import torch, monai, pydicom; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('monai', monai.__version__)"
echo ">> Done. Activate with: source $VENV_DIR/bin/activate"
