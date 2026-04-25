#!/usr/bin/env python3
"""
Fine-tune DeepFilterNet3 for AEC residual suppression.

This script fine-tunes a pre-trained DeepFilterNet3 model to additionally
remove robot voice residuals left by WebRTC AEC, while preserving its
existing ambient noise suppression capabilities.

Strategy:
    - Load pre-trained DeepFilterNet3 weights
    - Low learning rate to avoid catastrophic forgetting
    - Mixed dataset: 50% ambient noise + 50% robot residual scenarios
    - Train with the same loss functions as original DF (multi-resolution STFT loss)

Usage:
    python train.py \
        --dataset_dir data/dataset \
        --model_dir assets/downloads/DeepFilterNet3 \
        --output_dir checkpoints \
        --epochs 20 \
        --batch_size 4 \
        --lr 1e-5
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import _patch_torchaudio  # noqa: F401  — must come before df
import df as DF_MODULE
from df.enhance import df_features, get_device
from df.model import ModelParams


SAMPLE_RATE = 16000
RESAMPLE_FACTOR = 3  # 16kHz -> 48kHz for DeepFilterNet


class AECDataset(Dataset):
    """Dataset of (noisy, clean) pairs for DeepFilterNet fine-tuning."""

    def __init__(self, dataset_dir: str, split: str = "train"):
        self.noisy_dir = os.path.join(dataset_dir, split, "noisy")
        self.clean_dir = os.path.join(dataset_dir, split, "clean")

        self.files = sorted(glob.glob(os.path.join(self.noisy_dir, "*.wav")))
        if not self.files:
            raise FileNotFoundError(
                f"No WAV files found in {self.noisy_dir}. "
                "Run generate_dataset.py first."
            )
        print(f"  {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import soundfile as sf
        from scipy import signal

        fname = os.path.basename(self.files[idx])

        # Load 16kHz audio
        noisy_16k, _ = sf.read(
            os.path.join(self.noisy_dir, fname), dtype="float32"
        )
        clean_16k, _ = sf.read(
            os.path.join(self.clean_dir, fname), dtype="float32"
        )

        # Resample to 48kHz (DeepFilterNet's native rate)
        noisy_48k = signal.resample_poly(noisy_16k, RESAMPLE_FACTOR, 1).astype(
            np.float32
        )
        clean_48k = signal.resample_poly(clean_16k, RESAMPLE_FACTOR, 1).astype(
            np.float32
        )

        return (
            torch.from_numpy(noisy_48k),
            torch.from_numpy(clean_48k),
        )


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss used by DeepFilterNet.

    Combines spectral convergence loss and log-magnitude loss at
    multiple STFT resolutions for robust frequency-domain supervision.
    """

    def __init__(
        self,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_sizes=(512, 1024, 2048),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft(self, x, fft_size, hop_size, win_size):
        window = torch.hann_window(win_size, device=x.device)
        # Pad if needed
        if x.shape[-1] < win_size:
            x = F.pad(x, (0, win_size - x.shape[-1]))
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True,
        )
        return torch.abs(stft)

    def forward(self, predicted, target):
        loss = 0.0
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            pred_mag = self._stft(predicted, fft_size, hop_size, win_size)
            tgt_mag = self._stft(target, fft_size, hop_size, win_size)

            # Spectral convergence loss
            sc_loss = torch.norm(tgt_mag - pred_mag, p="fro") / (
                torch.norm(tgt_mag, p="fro") + 1e-8
            )

            # Log magnitude loss
            log_loss = F.l1_loss(
                torch.log1p(pred_mag), torch.log1p(tgt_mag)
            )

            loss += sc_loss + log_loss

        return loss / len(self.fft_sizes)


def _extract_features(audio_np, df_state, nb_df, device):
    """Extract DF features from numpy audio. Non-differentiable (input prep only)."""
    spec, erb_feat, spec_feat = df_features(
        torch.from_numpy(audio_np).unsqueeze(0),  # [1, T]
        df_state, nb_df, device=device,
    )
    return spec, erb_feat, spec_feat


def _spectral_loss(enhanced_spec, clean_spec):
    """L1 loss in complex spectral domain between model output and clean target."""
    # Both are real-valued tensors of shape [B, 1, T, F, 2] (re/im)
    return F.l1_loss(enhanced_spec, clean_spec)


def train_epoch(model, df_state, dataloader, optimizer, loss_fn, device, epoch):
    """Train for one epoch.

    Strategy: extract STFT features (non-differentiable numpy/C++ code),
    run the model forward (differentiable), compute loss in spectral domain.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_size = noisy.shape[0]

        for i in range(batch_size):
            noisy_np = noisy[i].numpy()
            clean_np = clean[i].numpy()

            # Extract features (non-differentiable — just STFT + normalization)
            spec, erb_feat, spec_feat = _extract_features(
                noisy_np, df_state, nb_df, device
            )

            # Get clean spectrum as target (non-differentiable)
            clean_spec_raw = df_state.analysis(clean_np[np.newaxis, :])  # [1, T, F] complex
            clean_spec = torch.as_tensor(
                np.stack([clean_spec_raw.real, clean_spec_raw.imag], axis=-1),
                dtype=torch.float32,
            ).unsqueeze(1).to(device)  # [1, 1, T, F, 2]

            # Reset hidden state
            if hasattr(model, "reset_h0"):
                model.reset_h0(batch_size=1, device=device)

            # Model forward — THIS is differentiable
            enhanced_spec = model(spec.clone(), erb_feat, spec_feat)[0]

            # Trim to same time dimension
            t_min = min(enhanced_spec.shape[2], clean_spec.shape[2])
            sample_loss = _spectral_loss(
                enhanced_spec[:, :, :t_min], clean_spec[:, :, :t_min]
            )
            batch_loss += sample_loss

        batch_loss = batch_loss / batch_size
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg = total_loss / num_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] loss={avg:.8f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, df_state, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))

    for noisy, clean in dataloader:
        batch_loss = 0.0
        batch_size = noisy.shape[0]

        for i in range(batch_size):
            noisy_np = noisy[i].numpy()
            clean_np = clean[i].numpy()

            spec, erb_feat, spec_feat = _extract_features(
                noisy_np, df_state, nb_df, device
            )
            clean_spec_raw = df_state.analysis(clean_np[np.newaxis, :])
            clean_spec = torch.as_tensor(
                np.stack([clean_spec_raw.real, clean_spec_raw.imag], axis=-1),
                dtype=torch.float32,
            ).unsqueeze(1).to(device)

            if hasattr(model, "reset_h0"):
                model.reset_h0(batch_size=1, device=device)

            enhanced_spec = model(spec.clone(), erb_feat, spec_feat)[0]

            t_min = min(enhanced_spec.shape[2], clean_spec.shape[2])
            sample_loss = _spectral_loss(
                enhanced_spec[:, :, :t_min], clean_spec[:, :, :t_min]
            )
            batch_loss += sample_loss

        batch_loss = batch_loss / batch_size
        total_loss += batch_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepFilterNet3 for AEC residual suppression"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="data/dataset",
        help="Path to generated dataset (from generate_dataset.py)",
    )
    parser.add_argument(
        "--model_dir", type=str, default="assets/downloads/DeepFilterNet3",
        help="Path to pre-trained DeepFilterNet3 model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints",
        help="Output directory for fine-tuned checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate (keep low to avoid catastrophic forgetting)",
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine",
        choices=["cosine", "constant"],
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save checkpoint every N epochs",
    )
    args = parser.parse_args()

    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}")

    # Load pre-trained DeepFilterNet3
    print(f"Loading DeepFilterNet3 from {args.model_dir}...")
    if not os.path.isdir(args.model_dir):
        print(
            f"ERROR: Model directory not found: {args.model_dir}\n"
            "Download DeepFilterNet3 first or provide the correct path."
        )
        return

    model, df_state, _ = DF_MODULE.init_df(model_base_dir=args.model_dir)
    if device == "cuda":
        model = model.to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Freeze encoder layers initially to preserve noise suppression knowledge
    # Only fine-tune the decoder and enhancement layers
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        # Keep all parameters trainable but with very low LR
        # The low LR + mixed dataset is the primary forgetting prevention
        param.requires_grad = True
        trainable_params.append(param)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Dataset
    print("Loading dataset...")
    train_dataset = AECDataset(args.dataset_dir, split="train")
    val_dataset = AECDataset(args.dataset_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Loss and optimizer
    loss_fn = MultiResolutionSTFTLoss().to(device)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    else:
        scheduler = None

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print(f"  LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, df_state, train_loader, optimizer, loss_fn, device, epoch
        )
        val_loss = validate(model, df_state, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.8f} | val_loss={val_loss:.8f} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
        })

        if scheduler:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_path)
            print(f"  -> Saved best model (val_loss={val_loss:.8f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # Save training history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.8f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(
        "\nTo use the fine-tuned model in the noise_cancellation node, "
        "load the checkpoint and replace the model weights."
    )


if __name__ == "__main__":
    main()
