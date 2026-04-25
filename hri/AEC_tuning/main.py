#!/usr/bin/env python3
"""
AEC Tuning Pipeline - Fine-tune DeepFilterNet3 for robot echo residual suppression.

This pipeline creates a DeepFilterNet3 model that can surgically remove
residual robot voice (left by WebRTC AEC) while preserving its existing
ambient noise suppression capabilities.

Audio flow in the robot:
    Mic → WebRTC AEC (removes gross echo) → DeepFilterNet (fine-tuned) → ASR
                                                    ↑
                                            Removes: ambient noise
                                                   + AEC residuals (robot voice leaks)

Steps:
    1. Generate Frida's voice library using Kokoro TTS
    2. Generate mixed training dataset (human speech + robot residual + ambient noise)
    3. Fine-tune DeepFilterNet3
    4. Export fine-tuned model for deployment

Usage:
    # Full pipeline:
    python main.py --step all

    # Or run steps individually:
    python main.py --step generate_voice
    python main.py --step generate_dataset
    python main.py --step train
    python main.py --step export
"""

import argparse
import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def run_step(script: str, args: list):
    """Run a sub-script with arguments."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)


def step_download_data(args):
    """Step 1: Download clean speech and noise datasets."""
    run_step("download_data.py", [
        "--output_dir", args.data_dir,
    ])


def step_generate_voice(args):
    """Step 2: Generate Frida's voice using Kokoro TTS."""
    run_step("generate_frida_voice.py", [
        "--output_dir", args.data_dir + "/frida_voice",
        "--voice", args.voice,
        "--speeds", "0.9,1.0,1.1",
    ])


def step_generate_dataset(args):
    """Step 2: Generate training dataset with mixed scenarios."""
    run_step("generate_dataset.py", [
        "--frida_dir", args.data_dir + "/frida_voice",
        "--clean_speech_dir", args.data_dir + "/clean_speech",
        "--noise_dir", args.data_dir + "/noise",
        "--output_dir", args.data_dir + "/dataset",
        "--num_train", str(args.num_train),
        "--num_val", str(args.num_val),
    ])


def step_train(args):
    """Step 3: Fine-tune DeepFilterNet3."""
    run_step("train.py", [
        "--dataset_dir", args.data_dir + "/dataset",
        "--model_dir", args.model_dir,
        "--output_dir", args.checkpoint_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ])


def step_export(args):
    """Step 4: Export fine-tuned model for deployment."""
    import torch

    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        print(f"ERROR: No checkpoint found at {best_ckpt}. Run training first.")
        sys.exit(1)

    export_dir = os.path.join(args.checkpoint_dir, "DeepFilterNet3_AEC")
    os.makedirs(export_dir, exist_ok=True)

    # Copy the original model config files
    for fname in os.listdir(args.model_dir):
        if fname.endswith((".ini", ".cfg", ".json", ".yaml", ".yml", ".onnx")):
            src = os.path.join(args.model_dir, fname)
            dst = os.path.join(export_dir, fname)
            shutil.copy2(src, dst)

    # Copy the fine-tuned weights
    shutil.copy2(best_ckpt, os.path.join(export_dir, "best_model.pt"))

    print(f"\nExported fine-tuned model to: {export_dir}")
    print("To use in the noise_cancellation node:")
    print(f"  1. Copy {export_dir} to your robot's assets/downloads/")
    print("  2. Update DF_MODEL_PATH parameter to point to the new model")
    print("  3. Load checkpoint weights after DF_MODULE.init_df()")


def main():
    parser = argparse.ArgumentParser(
        description="AEC Tuning Pipeline for DeepFilterNet3"
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "download_data", "generate_voice", "generate_dataset", "train", "export"],
        help="Which step to run",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join(SCRIPT_DIR, "data"),
        help="Base data directory",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default=os.path.join(SCRIPT_DIR, "assets", "downloads", "DeepFilterNet3"),
        help="Path to pre-trained DeepFilterNet3 model",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default=os.path.join(SCRIPT_DIR, "checkpoints"),
        help="Checkpoint output directory",
    )
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    print("AEC Tuning Pipeline for DeepFilterNet3")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    steps = {
        "download_data": step_download_data,
        "generate_voice": step_generate_voice,
        "generate_dataset": step_generate_dataset,
        "train": step_train,
        "export": step_export,
    }

    if args.step == "all":
        for name, func in steps.items():
            print(f"\n{'#'*60}")
            print(f"# Step: {name}")
            print(f"{'#'*60}")
            func(args)
    else:
        steps[args.step](args)

    print("\nDone!")


if __name__ == "__main__":
    main()
