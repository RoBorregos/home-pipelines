#!/usr/bin/env python3
"""
Generate training dataset for DeepFilterNet AEC fine-tuning.

Creates paired (noisy_input, clean_target) samples that simulate the
scenario after WebRTC AEC: the gross echo is removed, but residual
robot voice leaks through. DeepFilterNet learns to surgically remove
these residuals while preserving human speech.

Dataset structure:
    data/train/
        noisy/    - mic signal after WebRTC AEC (human + residual robot + ambient noise)
        clean/    - clean human speech only (target)
        ref/      - robot reference signal (what Frida said)
    data/val/
        noisy/
        clean/
        ref/

Usage:
    # First generate Frida's voice:
    python generate_frida_voice.py --output_dir data/frida_voice

    # Then download clean speech and noise datasets, and generate training pairs:
    python generate_dataset.py \
        --frida_dir data/frida_voice \
        --clean_speech_dir data/clean_speech \
        --noise_dir data/noise \
        --output_dir data/train \
        --num_samples 5000
"""

import argparse
import glob
import json
import os
import random
import sys
import wave

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 16000
CLIP_DURATION_S = 5.0  # Each training clip length
CLIP_SAMPLES = int(CLIP_DURATION_S * SAMPLE_RATE)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target rate. Returns float32 in [-1, 1]."""
    audio, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: str, sr: int = SAMPLE_RATE):
    """Save float32 audio as 16-bit WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def random_crop_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    """Randomly crop or zero-pad audio to exact length."""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length]
    else:
        padded = np.zeros(length, dtype=np.float32)
        start = random.randint(0, length - len(audio))
        padded[start:start + len(audio)] = audio
        return padded


def simulate_speaker_distortion(audio: np.ndarray) -> np.ndarray:
    """
    Simulate robot speaker + room acoustics distortion.

    The robot's speaker (typically a small speaker near the mic) introduces:
    - Frequency response coloring (small speaker = less bass, more mids)
    - Slight nonlinear distortion (clipping from cheap amp)
    - Room reflections
    """
    # Small speaker frequency response: attenuate bass, boost mids
    # Simple high-pass to simulate small speaker rolloff
    b_hp, a_hp = signal.butter(2, 300 / (SAMPLE_RATE / 2), btype="high")
    audio = signal.filtfilt(b_hp, a_hp, audio).astype(np.float32)

    # Slight mid-range resonance (typical of small enclosures)
    freq_peak = random.uniform(800, 2000)
    q = random.uniform(2, 5)
    b_peak, a_peak = signal.iirpeak(freq_peak / (SAMPLE_RATE / 2), q)
    gain_db = random.uniform(3, 8)
    gain = 10 ** (gain_db / 20)
    audio_peaked = signal.filtfilt(b_peak, a_peak, audio).astype(np.float32)
    audio = audio + (gain - 1) * audio_peaked

    # Slight nonlinear distortion (soft clipping)
    if random.random() < 0.3:
        clip_level = random.uniform(0.7, 0.95)
        audio = np.tanh(audio / clip_level) * clip_level

    return audio.astype(np.float32)


def simulate_aec_residual(
    robot_audio: np.ndarray,
    attenuation_db: float = None,
    add_comb_artifact: bool = None,
) -> np.ndarray:
    """
    Simulate what WebRTC AEC leaves behind after cancellation.

    WebRTC AEC removes ~20-40dB of echo but leaves residuals:
    - Attenuated echo (the filter doesn't converge perfectly)
    - Comb-filter artifacts from imperfect delay estimation
    - Nonlinear residual (distortion the linear filter can't model)
    """
    if attenuation_db is None:
        # WebRTC AEC typically achieves 20-40dB ERLE
        # Residual is what's left: -20 to -40 dB of original
        attenuation_db = random.uniform(20, 40)

    attenuation = 10 ** (-attenuation_db / 20)
    residual = robot_audio * attenuation

    # Comb-filter artifact (imperfect delay estimation)
    if add_comb_artifact is None:
        add_comb_artifact = random.random() < 0.4

    if add_comb_artifact:
        delay_samples = random.randint(1, 8)
        comb_gain = random.uniform(0.1, 0.4)
        delayed = np.zeros_like(residual)
        delayed[delay_samples:] = residual[:-delay_samples]
        residual = residual + comb_gain * delayed

    # Spectral coloring from imperfect adaptive filter
    if random.random() < 0.5:
        b_color, a_color = signal.butter(
            1, random.uniform(0.1, 0.4), btype="high"
        )
        residual = signal.filtfilt(b_color, a_color, residual).astype(np.float32)

    return residual.astype(np.float32)


def mix_at_snr(signal_audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal and noise at a given SNR (dB)."""
    sig_power = np.mean(signal_audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal_audio + noise * scale


def generate_sample(
    clean_speech_files: list,
    frida_voice_files: list,
    noise_files: list,
    include_ambient_noise: bool = True,
    include_robot_residual: bool = True,
) -> tuple:
    """
    Generate one training sample.

    Returns:
        (noisy_input, clean_target, reference_signal)
        - noisy_input: what the mic captures after WebRTC AEC
        - clean_target: the clean human speech (what DF should output)
        - reference_signal: what the robot said (for potential conditioning)
    """
    # 1. Load and crop clean human speech
    speech_file = random.choice(clean_speech_files)
    clean_speech = load_audio(speech_file)
    clean_speech = random_crop_or_pad(clean_speech, CLIP_SAMPLES)

    # Normalize speech
    peak = np.max(np.abs(clean_speech)) + 1e-10
    clean_speech = clean_speech / peak * random.uniform(0.3, 0.9)

    # Start building the noisy mix
    noisy = clean_speech.copy()

    # 2. Add robot voice residual (post-AEC)
    reference = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    if include_robot_residual and frida_voice_files:
        robot_file = random.choice(frida_voice_files)
        robot_audio = load_audio(robot_file)

        # Simulate speaker distortion (robot speaker characteristics)
        robot_distorted = simulate_speaker_distortion(robot_audio)

        # Random time offset (robot might start speaking before/after human)
        robot_padded = random_crop_or_pad(robot_distorted, CLIP_SAMPLES)

        # Simulate AEC residual (what's left after WebRTC cancels most of it)
        residual = simulate_aec_residual(robot_padded)

        # Mix residual with speech at a realistic level
        # Post-AEC residual is typically -5 to -20 dB relative to speech
        residual_snr = random.uniform(5, 25)
        noisy = mix_at_snr(noisy, residual, residual_snr)

        # Keep the clean robot audio as reference
        reference = random_crop_or_pad(robot_audio, CLIP_SAMPLES)

    # 3. Add ambient noise (the kind DeepFilterNet already handles)
    if include_ambient_noise and noise_files:
        noise_file = random.choice(noise_files)
        noise = load_audio(noise_file)
        noise = random_crop_or_pad(noise, CLIP_SAMPLES)

        ambient_snr = random.uniform(5, 30)
        noisy = mix_at_snr(noisy, noise, ambient_snr)

    # Clip to valid range
    noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)
    clean_speech = np.clip(clean_speech, -1.0, 1.0).astype(np.float32)

    return noisy, clean_speech, reference


def collect_audio_files(directory: str) -> list:
    """Recursively find all WAV/FLAC files in a directory."""
    if not os.path.isdir(directory):
        print(f"WARNING: Directory not found: {directory}")
        return []
    patterns = ["**/*.wav", "**/*.flac", "**/*.WAV", "**/*.FLAC"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AEC fine-tuning dataset for DeepFilterNet"
    )
    parser.add_argument(
        "--frida_dir", type=str, default="data/frida_voice",
        help="Directory with Frida's TTS audio (from generate_frida_voice.py)",
    )
    parser.add_argument(
        "--clean_speech_dir", type=str, default="data/clean_speech",
        help="Directory with clean human speech (e.g., LibriSpeech, DNS-Challenge clean)",
    )
    parser.add_argument(
        "--noise_dir", type=str, default="data/noise",
        help="Directory with ambient noise (e.g., MS-SNSD, DNS-Challenge noise)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/dataset",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--num_train", type=int, default=5000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_val", type=int, default=500,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Collect audio files
    print("Collecting audio files...")
    frida_files = collect_audio_files(args.frida_dir)
    speech_files = collect_audio_files(args.clean_speech_dir)
    noise_files = collect_audio_files(args.noise_dir)

    print(f"  Frida voice files: {len(frida_files)}")
    print(f"  Clean speech files: {len(speech_files)}")
    print(f"  Noise files: {len(noise_files)}")

    if not speech_files:
        print(
            "\nERROR: No clean speech files found. Download a dataset first:\n"
            "  python download_data.py --output_dir data\n"
            "  Or manually download:\n"
            "  - LibriSpeech: https://www.openslr.org/12\n"
            "  - DNS-Challenge clean: https://github.com/microsoft/DNS-Challenge\n"
            "  Place files in: {}\n".format(args.clean_speech_dir)
        )
        sys.exit(1)

    if not frida_files:
        print(
            "\nERROR: No Frida voice files found. Generate them first:\n"
            "  python generate_frida_voice.py --output_dir {}\n".format(args.frida_dir)
        )
        sys.exit(1)

    # Dataset composition:
    # 50% - robot residual + ambient noise + human speech  (new skill)
    # 25% - ambient noise only + human speech              (preserve existing skill)
    # 15% - robot residual only + human speech             (focused AEC cleanup)
    # 10% - clean speech only                              (identity mapping)
    compositions = [
        (0.50, True, True),   # both robot residual and ambient noise
        (0.25, True, False),  # ambient noise only (preserve DF's existing strength)
        (0.15, False, True),  # robot residual only
        (0.10, False, False), # clean speech only (no corruption)
    ]

    for split, num_samples in [("train", args.num_train), ("val", args.num_val)]:
        print(f"\nGenerating {split} split ({num_samples} samples)...")

        noisy_dir = os.path.join(args.output_dir, split, "noisy")
        clean_dir = os.path.join(args.output_dir, split, "clean")
        ref_dir = os.path.join(args.output_dir, split, "ref")
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)

        manifest = []
        sample_idx = 0

        for fraction, include_noise, include_robot in compositions:
            count = int(num_samples * fraction)
            label = []
            if include_noise:
                label.append("ambient")
            if include_robot:
                label.append("robot_residual")
            if not label:
                label.append("clean_only")
            label_str = "+".join(label)

            print(f"  {label_str}: {count} samples")

            for _ in range(count):
                noisy, clean, ref = generate_sample(
                    speech_files, frida_files, noise_files,
                    include_ambient_noise=include_noise,
                    include_robot_residual=include_robot,
                )

                fname = f"{sample_idx:06d}.wav"
                save_wav(noisy, os.path.join(noisy_dir, fname))
                save_wav(clean, os.path.join(clean_dir, fname))
                save_wav(ref, os.path.join(ref_dir, fname))

                manifest.append({
                    "id": sample_idx,
                    "file": fname,
                    "composition": label_str,
                    "duration_s": CLIP_DURATION_S,
                })

                sample_idx += 1
                if sample_idx % 100 == 0:
                    print(f"    {sample_idx}/{num_samples}")

        manifest_path = os.path.join(args.output_dir, split, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDataset generated at {args.output_dir}")
    print("Structure:")
    print(f"  {args.output_dir}/train/noisy/  - mic input (post-WebRTC AEC)")
    print(f"  {args.output_dir}/train/clean/  - target clean speech")
    print(f"  {args.output_dir}/train/ref/    - robot reference signal")
    print(f"  {args.output_dir}/val/...       - validation split")


if __name__ == "__main__":
    main()
