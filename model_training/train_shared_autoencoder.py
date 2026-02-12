#!/usr/bin/env python3
"""
Train a two-branch (RNA / Morphology) autoencoder with a shared latent space.

This script implements the shared autoencoder model described in:
[Cellular morphology emerges from polygenic, distributed transcriptional variation].

The model consists of:
- An RNA encoder
- A Morphology encoder
- A shared latent representation
- Modality-specific decoders

After training, morphology is predicted from RNA by passing RNA through
the RNA encoder and Morphology decoder.

This script is written for one dataset (e.g., LUAD). It can be reused for
other datasets (LINCS, TOARF, CRP) by modifying input/output paths or
passing them via CLI arguments.

Reproducibility details:
- RNA and morphology splits use identical sample indices to preserve alignment.
- Gradient clipping is applied once per optimizer update.
- All outputs (loss logs, model weights, predictions, metrics) are saved
  to the specified output directory.

Example:
  python train_shared_autoencoder.py \
      --dataset LUAD \
      --rna_csv /path/to/gene_clean_scaled.csv \
      --morph_csv /path/to/morph_decorrelated.csv \
      --out_dir /path/to/LUAD_results \
      --latent_dim 150 \
      --epochs 200 \
      --batch_size 16
"""


from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_numeric_csv(csv_path: str | Path) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Load a CSV into a numeric DataFrame and a float32 torch tensor."""
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    x = torch.tensor(df.values, dtype=torch.float32)
    return df, x


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_indices(
    n_samples: int,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx, test_idx) with val/test fractions.
    """
    if test_frac <= 0 or val_frac <= 0 or (test_frac + val_frac) >= 1:
        raise ValueError("Require: test_frac > 0, val_frac > 0, test_frac + val_frac < 1")

    all_idx = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        all_idx, test_size=(test_frac + val_frac), random_state=seed, shuffle=True
    )

    # proportion of temp that goes to test
    test_ratio_within_temp = test_frac / (test_frac + val_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio_within_temp, random_state=seed, shuffle=True
    )

    return train_idx, val_idx, test_idx


# -------------------------
# Model
# -------------------------
class SharedLatentAutoencoder(nn.Module):
    """
    Two independent autoencoders (RNA and Morphology) that share the latent dimension.
    """
    def __init__(self, rna_input_dim: int, morph_input_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_input_dim, latent_dim),
        )
        self.rna_decoder = nn.Sequential(
            nn.Linear(latent_dim, rna_input_dim),
        )

        self.morph_encoder = nn.Sequential(
            nn.Linear(morph_input_dim, latent_dim),
        )
        self.morph_decoder = nn.Sequential(
            nn.Linear(latent_dim, morph_input_dim),
        )

    def forward_rna(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.rna_encoder(x)
        x_hat = self.rna_decoder(z)
        return x_hat, z

    def forward_morph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.morph_encoder(x)
        x_hat = self.morph_decoder(z)
        return x_hat, z

    def encode_rna(self, x: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(x)

    def decode_morph(self, z: torch.Tensor) -> torch.Tensor:
        return self.morph_decoder(z)


def init_weights_xavier(m: nn.Module) -> None:
    """Xavier uniform init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -------------------------
# Training / Eval
# -------------------------
@torch.no_grad()
def eval_epoch(
    model: SharedLatentAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_rna = 0.0
    loss_morph = 0.0
    n_batches = 0

    for x_rna, x_morph in loader:
        x_rna = x_rna.to(device)
        x_morph = x_morph.to(device)

        xhat_rna, _ = model.forward_rna(x_rna)
        xhat_morph, _ = model.forward_morph(x_morph)

        loss_rna += criterion(xhat_rna, x_rna).item()
        loss_morph += criterion(xhat_morph, x_morph).item()
        n_batches += 1

    return {
        "rna": loss_rna / max(n_batches, 1),
        "morph": loss_morph / max(n_batches, 1),
    }


def train_epoch(
    model: SharedLatentAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer_rna: optim.Optimizer,
    optimizer_morph: optim.Optimizer,
    device: torch.device,
    clip_rna: float,
    clip_morph: float,
) -> Dict[str, float]:
    model.train()
    loss_rna = 0.0
    loss_morph = 0.0
    n_batches = 0

    for x_rna, x_morph in loader:
        x_rna = x_rna.to(device)
        x_morph = x_morph.to(device)

        # --- RNA branch update ---
        optimizer_rna.zero_grad(set_to_none=True)
        xhat_rna, _ = model.forward_rna(x_rna)
        l_rna = criterion(xhat_rna, x_rna)
        l_rna.backward()
        if clip_rna and clip_rna > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.rna_encoder.parameters()) + list(model.rna_decoder.parameters()),
                max_norm=clip_rna,
            )
        optimizer_rna.step()

        # --- Morphology branch update ---
        optimizer_morph.zero_grad(set_to_none=True)
        xhat_morph, _ = model.forward_morph(x_morph)
        l_morph = criterion(xhat_morph, x_morph)
        l_morph.backward()
        if clip_morph and clip_morph > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.morph_encoder.parameters()) + list(model.morph_decoder.parameters()),
                max_norm=clip_morph,
            )
        optimizer_morph.step()

        loss_rna += float(l_rna.item())
        loss_morph += float(l_morph.item())
        n_batches += 1

    return {
        "rna": loss_rna / max(n_batches, 1),
        "morph": loss_morph / max(n_batches, 1),
    }


# -------------------------
# Config + Main
# -------------------------
@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    rna_csv: str
    morph_csv: str
    out_dir: str

    latent_dim: int = 150
    epochs: int = 200
    batch_size: int = 16

    lr_rna: float = 1e-4
    lr_morph: float = 1e-4
    wd_rna: float = 1e-6
    wd_morph: float = 1e-4

    test_frac: float = 0.15
    val_frac: float = 0.15
    seed: int = 42

    clip_rna: float = 1.0
    clip_morph: float = 2.0

    device: str = "auto"  # "auto", "cpu", "cuda"


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train shared-latent autoencoder and predict morphology from RNA."
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., LUAD, LINCS, TOARF, CBPR).")
    p.add_argument("--rna_csv", required=True, help="Path to RNA CSV (rows=samples, cols=genes/features).")
    p.add_argument("--morph_csv", required=True, help="Path to morphology CSV (rows=samples, cols=features).")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    p.add_argument("--latent_dim", type=int, default=150)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--lr_rna", type=float, default=1e-4)
    p.add_argument("--lr_morph", type=float, default=1e-4)
    p.add_argument("--wd_rna", type=float, default=1e-6)
    p.add_argument("--wd_morph", type=float, default=1e-4)

    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clip_rna", type=float, default=1.0)
    p.add_argument("--clip_morph", type=float, default=2.0)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_dir(cfg.out_dir)

    set_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # ---- Load data ----
    rna_df, rna_tensor = load_numeric_csv(cfg.rna_csv)
    morph_df, morph_tensor = load_numeric_csv(cfg.morph_csv)

    if rna_tensor.shape[0] != morph_tensor.shape[0]:
        raise ValueError(
            f"Sample mismatch: RNA has {rna_tensor.shape[0]} rows, morphology has {morph_tensor.shape[0]} rows."
        )

    n_samples = rna_tensor.shape[0]
    train_idx, val_idx, test_idx = split_indices(
        n_samples=n_samples,
        test_frac=cfg.test_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    train_ds = TensorDataset(rna_tensor[train_idx], morph_tensor[train_idx])
    val_ds = TensorDataset(rna_tensor[val_idx], morph_tensor[val_idx])
    test_ds = TensorDataset(rna_tensor[test_idx], morph_tensor[test_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ---- Model ----
    model = SharedLatentAutoencoder(
        rna_input_dim=rna_tensor.shape[1],
        morph_input_dim=morph_tensor.shape[1],
        latent_dim=cfg.latent_dim,
    ).to(device)
    model.apply(init_weights_xavier)

    criterion = nn.MSELoss()

    optimizer_rna = optim.Adam(
        list(model.rna_encoder.parameters()) + list(model.rna_decoder.parameters()),
        lr=cfg.lr_rna,
        weight_decay=cfg.wd_rna,
    )
    optimizer_morph = optim.Adam(
        list(model.morph_encoder.parameters()) + list(model.morph_decoder.parameters()),
        lr=cfg.lr_morph,
        weight_decay=cfg.wd_morph,
    )

    # ---- Train ----
    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer_rna=optimizer_rna,
            optimizer_morph=optimizer_morph,
            device=device,
            clip_rna=cfg.clip_rna,
            clip_morph=cfg.clip_morph,
        )
        va = eval_epoch(model=model, loader=val_loader, criterion=criterion, device=device)

        row = {
            "epoch": epoch,
            "train_loss_rna": tr["rna"],
            "train_loss_morph": tr["morph"],
            "val_loss_rna": va["rna"],
            "val_loss_morph": va["morph"],
        }
        history.append(row)

        print(
            f"[{cfg.dataset}] Epoch {epoch:03d}/{cfg.epochs} | "
            f"train RNA {tr['rna']:.6f} morph {tr['morph']:.6f} | "
            f"val RNA {va['rna']:.6f} morph {va['morph']:.6f}"
        )

    # ---- Test ----
    te = eval_epoch(model=model, loader=test_loader, criterion=criterion, device=device)
    combined_test_loss = 0.5 * (te["rna"] + te["morph"])
    print(
        f"[{cfg.dataset}] Test | RNA {te['rna']:.6f} morph {te['morph']:.6f} "
        f"combined {combined_test_loss:.6f}"
    )

    # ---- Predict morphology from RNA (all samples) ----
    model.eval()
    with torch.no_grad():
        z_rna = model.encode_rna(rna_tensor.to(device))
        pred_morph = model.decode_morph(z_rna).cpu().numpy()

    pred_df = pd.DataFrame(pred_morph, columns=list(morph_df.columns))
    pred_path = out_dir / f"{cfg.dataset}_predicted_morphology.csv"
    pred_df.to_csv(pred_path, index=False)

    # ---- Metrics ----
    mse_all = mean_squared_error(morph_tensor.numpy().ravel(), pred_morph.ravel())
    metrics = {
        "dataset": cfg.dataset,
        "n_samples": int(n_samples),
        "rna_dim": int(rna_tensor.shape[1]),
        "morph_dim": int(morph_tensor.shape[1]),
        "latent_dim": int(cfg.latent_dim),
        "test_loss_rna": float(te["rna"]),
        "test_loss_morph": float(te["morph"]),
        "combined_test_loss": float(combined_test_loss),
        "mse_pred_vs_true_morph_all": float(mse_all),
        "device": str(device),
        "seed": int(cfg.seed),
    }

    # ---- Save artifacts ----
    history_df = pd.DataFrame(history)
    history_df["test_loss_rna"] = te["rna"]
    history_df["test_loss_morph"] = te["morph"]
    history_df["combined_test_loss"] = combined_test_loss
    history_df.to_csv(out_dir / f"{cfg.dataset}_loss_history.csv", index=False)

    with open(out_dir / f"{cfg.dataset}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / f"{cfg.dataset}_train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Save weights + minimal architecture params for re-loading
    ckpt = {
        "state_dict": model.state_dict(),
        "rna_input_dim": int(rna_tensor.shape[1]),
        "morph_input_dim": int(morph_tensor.shape[1]),
        "latent_dim": int(cfg.latent_dim),
        "config": asdict(cfg),
        "metrics": metrics,
    }
    torch.save(ckpt, out_dir / f"{cfg.dataset}_shared_autoencoder_ckpt.pt")

    print(f"[{cfg.dataset}] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
