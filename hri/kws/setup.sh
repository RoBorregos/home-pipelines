#!/usr/bin/env bash
# Idempotent one-time environment setup for the openWakeWord training pipeline.
#
# Creates an isolated venv at .venv/, installs pinned deps, clones
# piper-sample-generator under .third_party/, and downloads every dataset
# train.py needs into .data/.
#
# Safe to re-run: every step short-circuits if its output already exists.
#
# Requirements:
#   - python3.10 or python3.11 on PATH (the repo-level /Desktop/home2/venv
#     runs on 3.13 which is too new for several training deps, so we build a
#     dedicated venv here instead).
#   - git (to clone piper-sample-generator)
#   - ~10 GB free disk under this directory for the datasets.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV_DIR="$HERE/.venv"
THIRD_PARTY_DIR="$HERE/.third_party"
PIPER_DIR="$THIRD_PARTY_DIR/piper-sample-generator"

# ---- 1. Pick a supported Python interpreter --------------------------------
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: need python3.10, 3.11, or 3.12 on PATH (found none)." >&2
  echo "       Python 3.13 is too new for several training dependencies." >&2
  echo "       Install one of the supported versions and re-run setup.sh." >&2
  exit 1
fi
echo "[setup] using interpreter: $($PYTHON_BIN --version) ($(command -v "$PYTHON_BIN"))"

# ---- 2. Create venv if missing ---------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  # Guard: if the existing venv's activate script points to a different
  # directory (e.g. it was copied from another project), recreate it so
  # installs land in the right place.
  # Use || true so grep's exit-1-on-no-match doesn't kill the script under set -e.
  VENV_IN_ACTIVATE=$(grep 'VIRTUAL_ENV=' "$VENV_DIR/bin/activate" 2>/dev/null \
    | grep -v 'cygpath\|unset' | head -1 | sed 's/.*VIRTUAL_ENV=//; s/^"//; s/"$//' || true)
  if [[ "$VENV_IN_ACTIVATE" != "$VENV_DIR" ]]; then
    echo "[setup] existing venv activate script points to '$VENV_IN_ACTIVATE' instead of '$VENV_DIR' — recreating venv"
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Sanity-check: python must now resolve inside the venv.
ACTIVE_PYTHON="$(command -v python)"
if [[ "$ACTIVE_PYTHON" != "$VENV_DIR/bin/python"* ]]; then
  echo "ERROR: venv activation failed — 'python' is '$ACTIVE_PYTHON', expected inside $VENV_DIR" >&2
  exit 1
fi

python -m pip install --upgrade pip wheel setuptools

# ---- 3. Install Python deps -------------------------------------------------
# Torch wheel strategy:
#   NVIDIA GPU present → CUDA build from download.pytorch.org
#   macOS             → PyPI wheel is already CPU-only; no CDN needed
#   Linux / no GPU    → CPU-only build from download.pytorch.org
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] NVIDIA GPU detected, installing CUDA torch wheels"
  pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.* torchaudio==2.2.*
elif [[ "$(uname)" == "Darwin" ]]; then
  echo "[setup] macOS detected, installing torch wheels from PyPI (already CPU-only)"
  pip install torch==2.2.* torchaudio==2.2.*
else
  echo "[setup] no NVIDIA GPU, installing CPU torch wheels"
  pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.* torchaudio==2.2.*
fi

pip install -r "$HERE/requirements.txt"

# ---- piper-phonemize compatibility shim ------------------------------------
# piper-phonemize has no macOS Python wheel on PyPI.  piper-tts ships the same
# espeak-ng binding (espeakbridge.so) with a compatible Python API, so we drop
# a one-file shim into the venv's site-packages that satisfies the
#   from piper_phonemize import phonemize_espeak
# import used by piper-sample-generator/generate_samples.py.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "$SITE_PACKAGES/piper_phonemize.py" << 'SHIM_EOF'
"""piper-phonemize shim: delegates to piper-tts's bundled espeakbridge."""
from piper.phonemize_espeak import EspeakPhonemizer as _Pep

_phonemizer = None

def phonemize_espeak(text, voice):
    global _phonemizer
    if _phonemizer is None:
        _phonemizer = _Pep()
    return _phonemizer.phonemize(voice, text)
SHIM_EOF
echo "[setup] installed piper_phonemize compatibility shim at $SITE_PACKAGES/piper_phonemize.py"

# onnxruntime flavor depends on whether we have CUDA. onnxruntime and
# onnxruntime-gpu conflict if both are installed, so we handle them here
# instead of pinning in requirements.txt.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] installing onnxruntime-gpu (CUDA build)"
  pip uninstall -y onnxruntime >/dev/null 2>&1 || true
  pip install onnxruntime-gpu
else
  echo "[setup] installing onnxruntime (CPU build)"
  pip uninstall -y onnxruntime-gpu >/dev/null 2>&1 || true
  pip install onnxruntime
fi

# ---- 4. Clone piper-sample-generator ---------------------------------------
# Pin to v2.0.0 — it's the last release that exposes generate_samples.py at
# the repo root, which is what openwakeword's train.py imports. Later tags
# (v3.x) refactored the layout into a package and break train.py's
# `from generate_samples import generate_samples`.
PIPER_TAG="v2.0.0"
mkdir -p "$THIRD_PARTY_DIR"
if [[ ! -d "$PIPER_DIR/.git" ]]; then
  echo "[setup] cloning piper-sample-generator@$PIPER_TAG into $PIPER_DIR"
  git clone --depth 1 --branch "$PIPER_TAG" \
    https://github.com/rhasspy/piper-sample-generator.git "$PIPER_DIR"
else
  # If we already cloned (possibly the default branch), make sure we're on
  # the pinned tag. Idempotent — no-op when already on v2.0.0.
  current_tag="$(git -C "$PIPER_DIR" describe --tags --exact-match 2>/dev/null || echo '')"
  if [[ "$current_tag" != "$PIPER_TAG" ]]; then
    echo "[setup] piper-sample-generator on wrong ref ($current_tag); checking out $PIPER_TAG"
    git -C "$PIPER_DIR" fetch --tags --depth 1 origin "$PIPER_TAG"
    git -C "$PIPER_DIR" checkout "$PIPER_TAG"
  else
    echo "[setup] piper-sample-generator already on $PIPER_TAG, skipping"
  fi
fi

# Install piper-sample-generator's own requirements if it ships a file.
# Strip piper-phonemize: it has no macOS wheel on PyPI; our shim (written
# earlier) provides the same API via piper-tts's bundled espeakbridge.
if [[ -f "$PIPER_DIR/requirements.txt" ]]; then
  TMP_PIPER_REQ=$(mktemp)
  grep -v '^piper-phonemize' "$PIPER_DIR/requirements.txt" > "$TMP_PIPER_REQ"
  # webrtcvad has no ARM64 macOS wheel on PyPI — build from source using the
  # Xcode clang so the correct SDK is found (Homebrew LLVM picks the wrong one).
  if [[ "$(uname)" == "Darwin" ]]; then
    CC=$(xcrun --find clang) CXX=$(xcrun --find clang++) \
      SDKROOT=$(xcrun --show-sdk-path) MACOSX_DEPLOYMENT_TARGET=11.0 \
      pip install -r "$TMP_PIPER_REQ" --no-binary webrtcvad
  else
    pip install -r "$TMP_PIPER_REQ"
  fi
  rm -f "$TMP_PIPER_REQ"
fi

# v2.0.0 needs a pretrained model file at models/en_US-libritts_r-medium.pt
# (the notebook downloads it separately). Fetch it if missing.
PIPER_MODEL="$PIPER_DIR/models/en_US-libritts_r-medium.pt"
if [[ ! -f "$PIPER_MODEL" ]]; then
  echo "[setup] fetching piper LibriTTS-R medium model"
  mkdir -p "$PIPER_DIR/models"
  curl -L -o "$PIPER_MODEL" \
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
fi

# ---- 5. Fetch datasets ------------------------------------------------------
# Non-interactive when invoked from run.sh so CI / scripted runs don't block.
if [[ "${OWW_SETUP_NONINTERACTIVE:-0}" == "1" ]]; then
  OWW_DOWNLOAD_YES=1 python "$HERE/download_data.py"
else
  python "$HERE/download_data.py"
fi

echo "[setup] done. Activate the venv with:  source $VENV_DIR/bin/activate"
