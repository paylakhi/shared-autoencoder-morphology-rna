#!/usr/bin/env python3
"""
preprocess.py

General-purpose preprocessing for matched multimodal datasets
(e.g., gene expression + Cell Painting morphology) across multiple datasets.

What it does
------------
1) Load gene + morphology tables
2) Derive/standardize sample IDs (e.g., well IDs) from each table
3) Match samples across modalities (optionally drop duplicates)
4) Keep numeric features only
5) Clean (inf->nan, drop high-missing features, drop rows with any remaining NaNs)
6) Gene preprocessing (optional, can be skipped if already normalized):
     - CPM -> log1p
     - optional low-expression filter
     - standard scaling
     - optional variance threshold
7) Morphology preprocessing:
     - standard scaling
     - optional decorrelation (drop highly correlated features)
8) Save processed outputs + a short report

Example usage
-------------
python preprocess_multimodal.py \
  --dataset_name LUAD \
  --morph_path data/LUAD/CellPainting.csv \
  --gene_path  data/LUAD/L1000.csv \
  --out_dir results/LUAD \
  --morph_id_col Metadata_well_position \
  --gene_id_col id \
  --gene_id_from colon_last_token \
  --dedup_ids

For another dataset with different ID columns:
python preprocess_multimodal.py \
  --dataset_name LINCS \
  --morph_path data/LINCS/morph.csv \
  --gene_path  data/LINCS/gene.csv \
  --out_dir results/LINCS \
  --morph_id_col well \
  --gene_id_col well \
  --gene_id_from identity \
  --dedup_ids
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# -------------------------
# ID parsing strategies
# -------------------------
def id_identity(x: str) -> str:
    return "" if not isinstance(x, str) else x.strip().lower()


def id_colon_last_token(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return x.split(":")[-1].strip().lower()


def id_underscore_last_token(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return x.split("_")[-1].strip().lower()


ID_STRATEGIES = {
    "identity": id_identity,
    "colon_last_token": id_colon_last_token,
    "underscore_last_token": id_underscore_last_token,
}


# -------------------------
# Helpers
# -------------------------
def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def drop_high_missing_features(df: pd.DataFrame, max_missing_frac: float) -> pd.DataFrame:
    keep_cols = df.columns[df.isna().mean(axis=0) < max_missing_frac]
    return df.loc[:, keep_cols].copy()


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def standard_scale(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x = scaler.fit_transform(df.values)
    return x, scaler


def compute_cpm_log1p(gene_counts: pd.DataFrame) -> pd.DataFrame:
    row_sums = gene_counts.sum(axis=1).replace(0, np.nan)
    cpm = gene_counts.div(row_sums, axis=0) * 1e6
    cpm = cpm.fillna(0.0)
    return np.log1p(cpm)


def decorrelate_features(df: pd.DataFrame, corr_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    return df.drop(columns=to_drop), to_drop


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess matched gene+morphology data for any dataset.")

    # Required
    parser.add_argument("--dataset_name", type=str, required=True, help="Short dataset name (used in reports).")
    parser.add_argument("--morph_path", type=str, required=True, help="Path to morphology CSV/TSV.")
    parser.add_argument("--gene_path", type=str, required=True, help="Path to gene expression CSV/TSV.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")

    # File format
    parser.add_argument("--morph_sep", type=str, default=",", help="Separator for morphology file (, or \\t).")
    parser.add_argument("--gene_sep", type=str, default=",", help="Separator for gene file (, or \\t).")

    # ID columns and parsing
    parser.add_argument("--morph_id_col", type=str, required=True, help="ID column in morphology table.")
    parser.add_argument("--gene_id_col", type=str, required=True, help="ID column in gene table.")
    parser.add_argument(
        "--gene_id_from",
        type=str,
        default="identity",
        choices=sorted(ID_STRATEGIES.keys()),
        help="How to derive IDs from gene_id_col.",
    )
    parser.add_argument(
        "--morph_id_from",
        type=str,
        default="identity",
        choices=sorted(ID_STRATEGIES.keys()),
        help="How to derive IDs from morph_id_col.",
    )

    # Matching behavior
    parser.add_argument(
        "--dedup_ids",
        action="store_true",
        help="If set, drop duplicate IDs within each modality (keep first) before matching.",
    )

    # Cleaning thresholds
    parser.add_argument("--max_missing_frac", type=float, default=0.2, help="Drop features with >= this missing fraction.")

    # Gene preprocessing options
    parser.add_argument(
        "--gene_input_is_counts",
        action="store_true",
        help="If set, apply CPM->log1p before scaling (recommended when input is raw counts).",
    )
    parser.add_argument(
        "--min_mean_logcpm",
        type=float,
        default=5.0,
        help="After logCPM, keep genes with mean > this threshold. Set <=0 to disable.",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.01,
        help="VarianceThreshold on scaled gene features. Set <=0 to disable.",
    )

    # Morphology options
    parser.add_argument(
        "--morph_corr_threshold",
        type=float,
        default=0.9,
        help="Drop morph features with abs(corr) > threshold. Set <=0 to disable decorrelation.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose ID parsers
    gene_id_fn: Callable[[str], str] = ID_STRATEGIES[args.gene_id_from]
    morph_id_fn: Callable[[str], str] = ID_STRATEGIES[args.morph_id_from]

    # -------------------------
    # Load
    # -------------------------
    print(f"[INFO] Dataset={args.dataset_name}")
    morph = pd.read_csv(args.morph_path, sep=args.morph_sep)
    gene = pd.read_csv(args.gene_path, sep=args.gene_sep)

    if args.morph_id_col not in morph.columns:
        raise ValueError(f"Missing morph_id_col='{args.morph_id_col}' in morphology file.")
    if args.gene_id_col not in gene.columns:
        raise ValueError(f"Missing gene_id_col='{args.gene_id_col}' in gene file.")

    # -------------------------
    # Standardize IDs and match
    # -------------------------
    morph = morph.copy()
    gene = gene.copy()

    morph["_sample_id"] = morph[args.morph_id_col].apply(morph_id_fn)
    gene["_sample_id"] = gene[args.gene_id_col].apply(gene_id_fn)

    # Optional dedup before matching
    if args.dedup_ids:
        morph = morph.drop_duplicates(subset=["_sample_id"]).copy()
        gene = gene.drop_duplicates(subset=["_sample_id"]).copy()

    common_ids = sorted(set(morph["_sample_id"]) & set(gene["_sample_id"]))
    if len(common_ids) == 0:
        raise ValueError("No overlapping sample IDs between gene and morphology after ID parsing.")

    morph_m = morph[morph["_sample_id"].isin(common_ids)].copy()
    gene_m = gene[gene["_sample_id"].isin(common_ids)].copy()

    # Sort by sample_id to enforce consistent row order
    morph_m = morph_m.sort_values("_sample_id").reset_index(drop=True)
    gene_m = gene_m.sort_values("_sample_id").reset_index(drop=True)

    if not morph_m["_sample_id"].equals(gene_m["_sample_id"]):
        raise ValueError("Sample IDs do not align after sorting. Check ID parsing or duplicates.")

    print(f"[INFO] Matched samples: {morph_m.shape[0]}")

    # -------------------------
    # Numeric + cleaning
    # -------------------------
    morph_num = select_numeric(morph_m)
    gene_num = select_numeric(gene_m)

    morph_num = replace_inf_with_nan(morph_num)
    gene_num = replace_inf_with_nan(gene_num)

    morph_num = drop_high_missing_features(morph_num, args.max_missing_frac)
    gene_num = drop_high_missing_features(gene_num, args.max_missing_frac)

    combined = pd.concat([morph_num, gene_num], axis=1).dropna(axis=0).reset_index(drop=True)

    morph_clean = combined.iloc[:, : morph_num.shape[1]].copy()
    gene_clean = combined.iloc[:, morph_num.shape[1] :].copy()

    n = combined.shape[0]
    print(f"[INFO] After cleaning: n={n} | morph_features={morph_clean.shape[1]} | gene_features={gene_clean.shape[1]}")

    # -------------------------
    # Gene preprocessing
    # -------------------------
    if args.gene_input_is_counts:
        gene_proc = compute_cpm_log1p(gene_clean)
        if args.min_mean_logcpm and args.min_mean_logcpm > 0:
            keep = gene_proc.columns[gene_proc.mean(axis=0) > args.min_mean_logcpm]
            gene_proc = gene_proc.loc[:, keep].copy()
    else:
        # already normalized (e.g., L1000), just pass through
        gene_proc = gene_clean.copy()

    gene_scaled, _ = standard_scale(gene_proc)
    gene_feature_names = gene_proc.columns.to_list()

    if args.variance_threshold and args.variance_threshold > 0:
        selector = VarianceThreshold(threshold=args.variance_threshold)
        gene_scaled = selector.fit_transform(gene_scaled)
        gene_feature_names = list(np.array(gene_feature_names)[selector.get_support()])

    gene_scaled_df = pd.DataFrame(gene_scaled, columns=gene_feature_names)

    # -------------------------
    # Morphology preprocessing
    # -------------------------
    morph_scaled, _ = standard_scale(morph_clean)
    morph_scaled_df = pd.DataFrame(morph_scaled, columns=morph_clean.columns)

    dropped = []
    if args.morph_corr_threshold and args.morph_corr_threshold > 0:
        morph_final_df, dropped = decorrelate_features(morph_scaled_df, args.morph_corr_threshold)
    else:
        morph_final_df = morph_scaled_df

    # -------------------------
    # Save outputs
    # -------------------------
    out_gene = out_dir / f"{args.dataset_name}_gene_processed.csv"
    out_morph = out_dir / f"{args.dataset_name}_morph_processed.csv"
    out_ids = out_dir / f"{args.dataset_name}_matched_sample_ids.csv"
    out_report = out_dir / f"{args.dataset_name}_preprocessing_report.txt"

    gene_scaled_df.to_csv(out_gene, index=False)
    morph_final_df.to_csv(out_morph, index=False)
    pd.DataFrame({"sample_id": morph_m.loc[combined.index, "_sample_id"].values}).to_csv(out_ids, index=False)

    with open(out_report, "w") as f:
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write("=== Inputs ===\n")
        f.write(f"morph_path: {args.morph_path}\n")
        f.write(f"gene_path:  {args.gene_path}\n\n")
        f.write("=== Matching ===\n")
        f.write(f"matched_samples: {morph_m.shape[0]}\n")
        f.write(f"final_samples_after_dropna: {n}\n")
        f.write(f"dedup_ids: {args.dedup_ids}\n")
        f.write(f"morph_id_col: {args.morph_id_col} | morph_id_from: {args.morph_id_from}\n")
        f.write(f"gene_id_col: {args.gene_id_col} | gene_id_from: {args.gene_id_from}\n\n")
        f.write("=== Features ===\n")
        f.write(f"morph_features_after_missing_filter: {morph_num.shape[1]}\n")
        f.write(f"gene_features_after_missing_filter:  {gene_num.shape[1]}\n")
        f.write(f"gene_input_is_counts: {args.gene_input_is_counts}\n")
        f.write(f"gene_features_after_processing: {gene_scaled_df.shape[1]}\n")
        f.write(f"variance_threshold: {args.variance_threshold}\n")
        f.write(f"morph_corr_threshold: {args.morph_corr_threshold}\n")
        f.write(f"morph_features_dropped_by_corr: {len(dropped)}\n")

    print("[DONE] Saved:")
    print(f"  - {out_gene}")
    print(f"  - {out_morph}")
    print(f"  - {out_ids}")
    print(f"  - {out_report}")


if __name__ == "__main__":
    main()

