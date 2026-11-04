#!/usr/bin/env python3
"""Idempotent dataset downloader for the openWakeWord training pipeline.

Mirrors the recipe in openWakeWord's ``automatic_model_training.ipynb``.
Fetches everything train.py needs into ``.data/`` next to this file:

    mit_rirs/                       MIT room impulse responses (WAV, 16 kHz)
    audioset_16k/                   AudioSet bal_train09 shard (WAV, 16 kHz)
    negative_features_large.npy     ACAV100M pre-computed OWW features
    validation_set_features.npy     False-positive validation features
    oww_features/                   OWW feature extractor ONNX files

Any target that already exists on disk is skipped, so re-running ``run.sh``
after a crash is cheap. Prints a size summary before starting so the user
can abort on a metered link.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".data"

# Approximate disk footprint for the summary (printed before anything runs).
APPROX_SIZES_GB = {
    "mit_rirs": 0.05,
    # ~0.68 GB parquet shard (retained under .data/audioset/) + ~1 GB of
    # extracted 16 kHz WAVs.
    "audioset_16k": 1.7,
    "negative_features_large.npy": 2.0,
    "validation_set_features.npy": 0.3,
    "oww_features": 0.05,
}


def _nonempty_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for _ in p.iterdir():
        return True
    return False


_DOWNLOAD_TIMEOUT_S = 30
_DOWNLOAD_MAX_RETRIES = 5


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"  fetching {url}")
    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        req = urllib.request.Request(url)
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as response:
                mode = "ab" if resume_from and response.status == 206 else "wb"
                with open(tmp_path, mode) as fh:
                    shutil.copyfileobj(response, fh)
            break
        except (OSError, TimeoutError) as exc:
            if attempt == _DOWNLOAD_MAX_RETRIES:
                raise
            print(f"  download attempt {attempt} failed ({exc}), retrying...")
    tmp_path.rename(out_path)


# --- Individual fetchers -----------------------------------------------------


def fetch_mit_rirs() -> None:
    """Stream the MIT RIR dataset and write 16 kHz WAVs to .data/mit_rirs/."""
    out_dir = DATA_DIR / "mit_rirs"
    if _nonempty_dir(out_dir):
        print(f"[skip] mit_rirs (already present at {out_dir})")
        return

    print("[get ] MIT Room Impulse Responses")
    import datasets
    import numpy as np
    import scipy.io.wavfile
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    for row in tqdm(rir_dataset, desc="mit_rirs"):
        name = row["audio"]["path"].split("/")[-1]
        if not name.lower().endswith(".wav"):
            name = os.path.splitext(name)[0] + ".wav"
        scipy.io.wavfile.write(
            out_dir / name,
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )


def fetch_audioset() -> None:
    """Download one AudioSet parquet shard and extract 16 kHz WAVs.

    The ``agkphysics/AudioSet`` dataset was converted from per-shard ``.tar``
    archives to parquet in 2024. We download a single balanced-train parquet
    file (~680 MB) and decode each embedded audio blob into a 16 kHz WAV in
    ``.data/audioset_16k/``. One parquet file yields ~1000 clips, which is
    plenty of background audio for training.
    """
    out_dir = DATA_DIR / "audioset_16k"
    if _nonempty_dir(out_dir):
        print(f"[skip] audioset_16k (already present at {out_dir})")
        return

    print("[get ] AudioSet (bal_train/09.parquet shard)")
    import io

    import numpy as np
    import pyarrow.parquet as pq
    import scipy.io.wavfile
    import scipy.signal
    import soundfile as sf
    from tqdm import tqdm

    parquet_dir = DATA_DIR / "audioset"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "bal_train_09.parquet"
    url = (
        "https://huggingface.co/datasets/agkphysics/AudioSet/"
        "resolve/main/data/bal_train/09.parquet"
    )
    if not parquet_path.exists():
        _download_file(url, parquet_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Figure out which column holds the audio blob. agkphysics/AudioSet stores
    # it as a struct column named "audio" with "bytes" + "path" fields.
    parquet_file = pq.ParquetFile(parquet_path)
    if "audio" not in parquet_file.schema_arrow.names:
        raise RuntimeError(
            f"Expected an 'audio' column in {parquet_path}, got "
            f"{parquet_file.schema_arrow.names}"
        )

    target_sr = 16000
    written = 0
    i = 0
    # Stream row-group batches instead of pq.read_table(...).to_pylist(), which
    # would materialize the whole ~680 MB shard's decoded audio blobs at once.
    for batch in tqdm(
        parquet_file.iter_batches(columns=["audio"], batch_size=256),
        desc="audioset_16k",
    ):
        for entry in batch.column("audio").to_pylist():
            if entry is None:
                i += 1
                continue
            audio_bytes = entry.get("bytes") if isinstance(entry, dict) else None
            path = (
                entry.get("path") if isinstance(entry, dict) else None
            ) or f"clip_{i:05d}"
            if not audio_bytes:
                i += 1
                continue
            try:
                data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            except Exception:
                i += 1
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                n_out = int(round(len(data) * target_sr / sr))
                if n_out <= 0:
                    i += 1
                    continue
                data = scipy.signal.resample(data, n_out)
            name = os.path.basename(path)
            name = os.path.splitext(name)[0] + ".wav"
            if not name or name == ".wav":
                name = f"clip_{i:05d}.wav"
            scipy.io.wavfile.write(
                out_dir / name,
                target_sr,
                (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16),
            )
            written += 1
            i += 1

    if written == 0:
        raise RuntimeError(
            f"No audio clips extracted from {parquet_path}; parquet schema may have changed."
        )
    print(f"  wrote {written} clips to {out_dir}")


def fetch_acav100m_features() -> None:
    out_path = DATA_DIR / "negative_features_large.npy"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] ACAV100M features (already present at {out_path})")
        return
    print("[get ] ACAV100M adversarial negative features")
    _download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        out_path,
    )


def fetch_validation_features() -> None:
    out_path = DATA_DIR / "validation_set_features.npy"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] validation features (already present at {out_path})")
        return
    print("[get ] False-positive validation features")
    _download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
        "resolve/main/validation_set_features.npy",
        out_path,
    )


def fetch_oww_feature_extractors() -> None:
    """Ensure openwakeword's melspectrogram/embedding ONNX files are cached.

    openwakeword ships a helper that downloads these on first use; calling it
    here keeps training runs self-contained and avoids a surprise download
    the first time train.py imports the package. download_models() checks
    per-file existence in the openwakeword package directory itself (not
    under .data/), so it's already a no-op on repeat calls — we don't need
    our own marker file to track that.
    """
    print("[get ] openwakeword feature extractors (mel + embedding)")
    try:
        import openwakeword.utils as utils

        utils.download_models()
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: openwakeword.download_models() failed: {exc}")
        print("  training will attempt to fetch them itself on first run.")


# --- Driver ------------------------------------------------------------------


FETCHERS = [
    ("mit_rirs", fetch_mit_rirs),
    ("audioset_16k", fetch_audioset),
    ("negative_features_large.npy", fetch_acav100m_features),
    ("validation_set_features.npy", fetch_validation_features),
    ("oww_features", fetch_oww_feature_extractors),
]


def _pending() -> list[str]:
    pending = []
    for name, _ in FETCHERS:
        p = DATA_DIR / name
        if name.endswith(".npy"):
            if not (p.exists() and p.stat().st_size > 0):
                pending.append(name)
        elif name == "oww_features":
            try:
                import openwakeword

                models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
                if not _nonempty_dir(models_dir):
                    pending.append(name)
            except ImportError:
                pending.append(name)
        else:
            if not _nonempty_dir(p):
                pending.append(name)
    return pending


def _print_summary() -> None:
    pending = _pending()
    if not pending:
        print("All datasets already present. Nothing to download.")
        return
    total = sum(APPROX_SIZES_GB.get(n, 0.0) for n in pending)
    print("The following assets will be downloaded:")
    for n in pending:
        print(f"  - {n:<40} ~{APPROX_SIZES_GB.get(n, 0.0):>5.2f} GB")
    print(f"Estimated total: ~{total:.1f} GB")
    print(f"Target directory: {DATA_DIR}")
    if os.environ.get("OWW_DOWNLOAD_YES") != "1":
        try:
            reply = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            print(
                "\nNo TTY to confirm on. Re-run with OWW_DOWNLOAD_YES=1 to "
                "download non-interactively."
            )
            sys.exit(1)
        if reply not in ("y", "yes"):
            print("Aborted.")
            sys.exit(1)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _print_summary()
    for name, fn in FETCHERS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - surface any download error
            print(f"ERROR while fetching {name}: {exc}", file=sys.stderr)
            return 1
    print("All datasets ready under", DATA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
