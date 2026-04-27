#!/usr/bin/env python3
"""
Download public datasets needed for AEC fine-tuning.

Downloads:
    - Clean speech: LibriSpeech dev-clean (for human speech targets)
    - Noise: MS-SNSD (Microsoft Scalable Noisy Speech Dataset)

Usage:
    python download_data.py --output_dir data
"""

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile


def download_file(url: str, dest: str):
    """Download a file using curl or wget."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")

    # Try curl first, then wget
    try:
        subprocess.run(
            ["curl", "-L", "-o", dest, "--progress-bar", url],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(
            ["wget", "-O", dest, "--show-progress", url],
            check=True,
        )


def download_librispeech_dev_clean(output_dir: str):
    """Download LibriSpeech dev-clean subset (~350MB, ~5.4h of speech)."""
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    archive = os.path.join(output_dir, "dev-clean.tar.gz")
    extract_dir = os.path.join(output_dir, "clean_speech")

    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print("  LibriSpeech dev-clean already extracted.")
        return

    download_file(url, archive)

    print("  Extracting...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Move FLAC files up from nested structure
    # LibriSpeech structure: LibriSpeech/dev-clean/speaker/chapter/file.flac
    print("  Flattening directory structure...")
    import glob
    flac_files = glob.glob(
        os.path.join(extract_dir, "**", "*.flac"), recursive=True
    )
    flat_dir = os.path.join(extract_dir, "flat")
    os.makedirs(flat_dir, exist_ok=True)
    for f in flac_files:
        dest = os.path.join(flat_dir, os.path.basename(f))
        if not os.path.exists(dest):
            os.rename(f, dest)

    print(f"  Done. {len(flac_files)} files in {flat_dir}")


def download_deepfilternet3(output_dir: str):
    """Download pre-trained DeepFilterNet3 model (~8MB)."""
    url = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip"
    model_dir = os.path.join(output_dir, "DeepFilterNet3")
    archive = os.path.join(output_dir, "DeepFilterNet3.zip")

    if os.path.isdir(model_dir) and os.listdir(model_dir):
        print("  DeepFilterNet3 model already exists.")
        return

    download_file(url, archive)

    print("  Extracting...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(output_dir)
    os.remove(archive)

    print(f"  Done. Model in {model_dir}")


def download_ms_snsd(output_dir: str):
    """Download MS-SNSD noise dataset."""
    url = "https://github.com/microsoft/MS-SNSD/archive/refs/heads/master.zip"
    archive = os.path.join(output_dir, "ms-snsd.zip")
    extract_dir = os.path.join(output_dir, "noise")

    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print("  MS-SNSD already extracted.")
        return

    download_file(url, archive)

    print("  Extracting...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)

    print(f"  Done. Noise files in {extract_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for AEC tuning")
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Base output directory",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Output directory for DeepFilterNet3 model (default: <script_dir>/assets/downloads)",
    )
    parser.add_argument(
        "--skip_speech", action="store_true",
        help="Skip downloading clean speech dataset",
    )
    parser.add_argument(
        "--skip_noise", action="store_true",
        help="Skip downloading noise dataset",
    )
    parser.add_argument(
        "--skip_model", action="store_true",
        help="Skip downloading DeepFilterNet3 model",
    )
    args = parser.parse_args()

    # Default model_dir next to this script: AEC_tuning/assets/downloads/
    if args.model_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args.model_dir = os.path.join(script_dir, "assets", "downloads")

    print("Downloading datasets and model for AEC fine-tuning...\n")

    if not args.skip_model:
        print("[1/3] DeepFilterNet3 pre-trained model")
        download_deepfilternet3(args.model_dir)

    if not args.skip_speech:
        print("\n[2/3] LibriSpeech dev-clean (clean speech targets)")
        download_librispeech_dev_clean(args.output_dir)

    if not args.skip_noise:
        print("\n[3/3] MS-SNSD (ambient noise)")
        download_ms_snsd(args.output_dir)

    print("\n" + "=" * 50)
    print("Downloads complete!")
    print(f"  Model: {args.model_dir}/DeepFilterNet3/")
    print(f"  Clean speech: {args.output_dir}/clean_speech/")
    print(f"  Noise: {args.output_dir}/noise/")
    print("\nNext steps:")
    print("  1. Generate Frida's voice:  python generate_frida_voice.py")
    print("  2. Generate dataset:        python generate_dataset.py")
    print("  3. Train:                   python train.py")
    print("  Or run all:                 python main.py --step all")


if __name__ == "__main__":
    main()
