#!/usr/bin/env python3
"""
Apply LUAD-trained shared autoencoder to donor-level data (no retraining).

This script reproduces the "Generalization to Individual-Level Data" step where a model
trained on LUAD perturbation data is applied (without retraining) to donor-level RNA-seq
and morphology data after harmonizing inputs (genes/features intersection).

It performs:
- Prediction: RNA encoder -> shared latent -> Morphology decoder
- Global evaluation: R² and MSE per morphology feature across donors
- Individual evaluation: per-donor squared error per feature (MSE per donor per feature),
  then mean/median across donors per feature

Outputs are written to --out_dir.

Example:
  python apply_to_donor_data.py \
    --ckpt /path/to/luad_ckpt.pt \
    --donor_rna_csv /path/to/donor_rna.csv \
    --donor_morph_csv /path/to/donor_morph.csv \
    --latent_dim 150 \
    --gene_list /path/to/genes_intersection_with_L1000.txt \
    --morph_feature_list /path/to/shared_morph_features.txt \
    --out_dir /path/to/out \
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# Model definition (matches the minimal Linear-only architecture)
# =============================================================================
class SharedAutoencoder(nn.Module):
    def __init__(self, rna_input_dim: int, morph_input_dim: int, latent_dim: int):
        super().__init__()
        self.rna_encoder = nn.Sequential(nn.Linear(rna_input_dim, latent_dim))
        self.morphology_encoder = nn.Sequential(nn.Linear(morph_input_dim, latent_dim))
        self.rna_decoder = nn.Sequential(nn.Linear(latent_dim, rna_input_dim))
        self.morphology_decoder = nn.Sequential(nn.Linear(latent_dim, morph_input_dim))

    def forward_rna(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.rna_encoder(x)
        x_hat = self.rna_decoder(z)
        return x_hat, z

    def forward_morphology(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.morphology_encoder(x)
        x_hat = self.morphology_decoder(z)
        return x_hat, z


# =============================================================================
# Evaluation helpers (self-contained; no extra files needed)
# =============================================================================
def r2_per_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    R² computed independently per feature (column), across donors (rows):
      R² = 1 - SSE/SST
    Returns NaN for features with zero variance (SST=0).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean_true = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean_true) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
        r2 = np.where(ss_tot == 0.0, np.nan, r2)
    return r2


def mse_per_feature(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MSE per feature across donors."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return np.mean((y_true - y_pred) ** 2, axis=0)


def per_donor_squared_error(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-donor squared error per feature.
    (For a single donor and single feature, this is (error^2).)
    """
    common = y_true_df.index.intersection(y_pred_df.index)
    y_true2 = y_true_df.loc[common]
    y_pred2 = y_pred_df.loc[common]
    se = (y_true2 - y_pred2) ** 2
    se.index.name = "donor"
    return se


# =============================================================================
# IO + harmonization helpers
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_list_file(path: str, uppercase: bool = False) -> List[str]:
    items: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s.upper() if uppercase else s)
    return items


def detect_id_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_table_with_id(path: str, id_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Load CSV and set the donor/sample identifier as index.
    If id_col is None, try to auto-detect. If none found, create synthetic IDs.
    """
    df = pd.read_csv(path)

    if id_col is None:
        id_col = detect_id_col(df, ["donor", "Donor", "sample", "sample_id", "IID", "iid", "id", "ID"])

    if id_col is None:
        df = df.copy()
        df.insert(0, "donor", [f"sample_{i}" for i in range(len(df))])
        id_col = "donor"

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)
    return df, id_col


def subset_and_align(
    rna_df: pd.DataFrame,
    morph_df: pd.DataFrame,
    gene_list: Optional[List[str]] = None,
    morph_feature_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align donors and optionally subset genes/features using provided lists.
    Numeric coercion + missing fill(0.0) is applied (matches common preprocessing expectations).
    """
    common = rna_df.index.intersection(morph_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping donors between donor RNA and donor morphology tables.")
    rna_df = rna_df.loc[common].copy()
    morph_df = morph_df.loc[common].copy()

    # numeric coercion
    rna_df = rna_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    morph_df = morph_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if gene_list is not None:
        keep_genes = [g for g in gene_list if g in rna_df.columns]
        if len(keep_genes) == 0:
            raise ValueError("After applying gene_list, no genes remain in donor RNA table.")
        rna_df = rna_df.loc[:, keep_genes]

    if morph_feature_list is not None:
        keep_feats = [m for m in morph_feature_list if m in morph_df.columns]
        if len(keep_feats) == 0:
            raise ValueError("After applying morph_feature_list, no features remain in donor morphology table.")
        morph_df = morph_df.loc[:, keep_feats]

    return rna_df, morph_df


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply LUAD-trained shared autoencoder to donor-level data (no retraining)."
    )

    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt/.pth).")
    p.add_argument("--out_dir", required=True, help="Output directory.")

    p.add_argument("--donor_rna_csv", required=True, help="Donor RNA CSV (rows=donors, cols=genes).")
    p.add_argument("--donor_morph_csv", required=True, help="Donor morphology CSV (rows=donors, cols=features).")

    p.add_argument("--donor_id_col", default=None, help="Donor ID column name if present (default: auto-detect).")

    p.add_argument(
        "--gene_list",
        default=None,
        help="Optional text file listing genes to keep (one per line).",
    )
    p.add_argument(
        "--morph_feature_list",
        default=None,
        help="Optional text file listing morphology features to keep (one per line).",
    )

    p.add_argument("--latent_dim", type=int, required=True, help="Latent dimension used in training (must match ckpt).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference.")

    p.add_argument(
        "--load_full_model",
        action="store_true",
        help="If set, --ckpt is a full torch model object (torch.save(model)). Otherwise expects a state_dict.",
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # Save config for reproducibility
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load donor data
    rna_df, _ = load_table_with_id(args.donor_rna_csv, args.donor_id_col)
    morph_df, _ = load_table_with_id(args.donor_morph_csv, args.donor_id_col)

    # Optional gene/feature lists
    gene_list = read_list_file(args.gene_list) if args.gene_list else None
    morph_feature_list = read_list_file(args.morph_feature_list) if args.morph_feature_list else None

    # Align + subset
    rna_df, morph_df = subset_and_align(rna_df, morph_df, gene_list, morph_feature_list)

    used_genes = list(rna_df.columns)
    used_feats = list(morph_df.columns)

    with open(os.path.join(args.out_dir, "genes_used.txt"), "w") as f:
        f.write("\n".join(used_genes) + "\n")
    with open(os.path.join(args.out_dir, "morph_features_used.txt"), "w") as f:
        f.write("\n".join(used_feats) + "\n")

    # Build/load model
    device = torch.device(args.device)

    rna_dim = rna_df.shape[1]
    morph_dim = morph_df.shape[1]

    if args.load_full_model:
        model = torch.load(args.ckpt, map_location=device)
        model.eval()
    else:
        model = SharedAutoencoder(rna_dim, morph_dim, args.latent_dim).to(device)
        state = torch.load(args.ckpt, map_location=device)

        # Allow wrapper checkpoints
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=True)
        model.eval()

    # Predict morphology from RNA
    x = torch.tensor(rna_df.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.rna_encoder(x)
        yhat = model.morphology_decoder(z)

    pred_df = pd.DataFrame(yhat.detach().cpu().numpy(), index=rna_df.index, columns=used_feats)
    true_df = morph_df.loc[pred_df.index, used_feats].copy()

    # Save predictions
    pred_df.to_csv(os.path.join(args.out_dir, "predicted_morphology_global.csv"))

    # Global evaluation (across donors)
    r2 = r2_per_feature(true_df.values, pred_df.values)
    mse = mse_per_feature(true_df.values, pred_df.values)

    global_r2 = pd.Series(r2, index=used_feats, name="r2")
    global_mse = pd.Series(mse, index=used_feats, name="mse")

    global_r2.to_csv(os.path.join(args.out_dir, "global_r2_per_feature.csv"), header=True)
    global_mse.to_csv(os.path.join(args.out_dir, "global_mse_per_feature.csv"), header=True)

    # Individual evaluation (per donor squared error per feature)
    se_df = per_donor_squared_error(true_df, pred_df)  # rows=donor, cols=feature
    se_df.to_csv(os.path.join(args.out_dir, "individual_per_donor_squared_error.csv"))

    individual_mean_mse = se_df.mean(axis=0).rename("mean_mse")
    individual_median_mse = se_df.median(axis=0).rename("median_mse")

    individual_mean_mse.to_csv(os.path.join(args.out_dir, "individual_mean_mse_per_feature.csv"), header=True)
    individual_median_mse.to_csv(os.path.join(args.out_dir, "individual_median_mse_per_feature.csv"), header=True)

    # Convenience combined table (useful for Figure 3 comparisons)
    combined = pd.DataFrame(
        {
            "global_r2": global_r2,
            "global_mse": global_mse,
            "individual_mean_mse": individual_mean_mse,
            "individual_median_mse": individual_median_mse,
        }
    )
    combined.to_csv(os.path.join(args.out_dir, "global_vs_individual_metrics_per_feature.csv"))

    # Console summary
    mean_r2 = float(np.nanmean(global_r2.values))
    mean_mse_over_features = float(np.nanmean(global_mse.values))
    mean_indiv_mse_over_features = float(np.nanmean(individual_mean_mse.values))

    print(f"[Global]    n_donors={len(true_df)}  mean_R2={mean_r2:.4f}  mean_MSE={mean_mse_over_features:.6f}")
    print(f"[Individual] n_donors={len(true_df)}  mean_feature_MSE={mean_indiv_mse_over_features:.6f}")
    print(f"[DONE] Results written to: {args.out_dir}")


if __name__ == "__main__":
    main()
