#!/usr/bin/env python3
"""
cmTWAS (eQTL-restricted expression–morphology association scan)

For each morphology feature (treated as a phenotype), this script tests:
    morphology_trait ~ gene_expression (+ optional covariates)

The scan is restricted to genes that are significant eQTL genes (based on an
external eQTL results file), reducing the search space.

Outputs:
- Per-trait sumstats TSV:
    cmTWAS_<TRAIT>.sumstats.tsv
  containing: trait, gene, beta, se, t, p, q, n, df

- Summary TSV:
    cmTWAS_summary_per_trait.tsv

- Diagnostic histogram:
    hist_minp_per_trait.png

- Top hits table:
    cmTWAS_top_hits_table.tsv

Example:
  python run_cmTWAS_eqtl_restricted.py \
    --morph_csv /path/morph.csv \
    --expr_csv  /path/expr.csv \
    --eqtl_tsv  /path/eqtl_results.txt \
    --out_dir   /path/out \
    --eqtl_p_cutoff 0.05 \
    --traits_csv /path/top_traits.csv \
    --use_traits_csv \
    --covar_num sex PC1 PC2 PC3 \
    --covar_cat plate batch
"""

from __future__ import annotations

import argparse
import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_donor_col(df: pd.DataFrame, donor_col_candidates: List[str] | None = None) -> pd.DataFrame:
    """Rename a donor/sample identifier column to 'donor'."""
    if donor_col_candidates is None:
        donor_col_candidates = ["donor", "IID", "iid", "sample", "sample_id", "Donor", "ID"]

    df = df.copy()
    for c in donor_col_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "donor"})
            break
    if "donor" not in df.columns:
        raise ValueError("Could not find a donor/sample identifier column. "
                         "Please ensure one column is named 'donor' (or provide a candidate name).")

    df["donor"] = df["donor"].astype(str)
    return df


def upper_gene_cols(expr: pd.DataFrame) -> pd.DataFrame:
    """Uppercase gene feature columns (keeps 'donor' unchanged)."""
    expr = expr.copy()
    expr.columns = [c if c == "donor" else str(c).strip().upper() for c in expr.columns]
    return expr


def build_covariate_matrix(
    morph_df: pd.DataFrame,
    covar_numeric: List[str],
    covar_categorical: List[str],
) -> pd.DataFrame:
    """
    Build covariate matrix with intercept.
    - numeric covariates: coerced to numeric
    - categorical covariates: one-hot (drop_first=True)
    - missing values: mean-imputed (simple, transparent choice)
    """
    parts = []

    if covar_numeric:
        missing = [c for c in covar_numeric if c not in morph_df.columns]
        if missing:
            raise ValueError(f"Missing numeric covariates in morph_df: {missing}")
        C_num = morph_df[covar_numeric].copy()
        for c in C_num.columns:
            C_num[c] = pd.to_numeric(C_num[c], errors="coerce")
        parts.append(C_num)

    if covar_categorical:
        missing = [c for c in covar_categorical if c not in morph_df.columns]
        if missing:
            raise ValueError(f"Missing categorical covariates in morph_df: {missing}")
        C_cat = pd.get_dummies(morph_df[covar_categorical].astype(str), drop_first=True)
        parts.append(C_cat)

    if parts:
        C = pd.concat(parts, axis=1)
    else:
        C = pd.DataFrame(index=morph_df.index)

    # intercept
    C.insert(0, "intercept", 1.0)

    # drop all-missing columns, then mean-impute remaining missing
    C = C.loc[:, ~(C.isna().all())]
    for c in C.columns:
        if C[c].isna().any():
            C[c] = C[c].fillna(C[c].mean())

    return C.astype(float)


def residualize_matrix(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Residualize X on covariates C via projection:
        X_resid = X - C (pinv(C) X)
    """
    C_pinv = np.linalg.pinv(C)
    return X - C @ (C_pinv @ X)


def assoc_scan_y_vs_manyX(y: np.ndarray, X: np.ndarray, df_resid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast univariate regression for many predictors:
      y ~ X_j  for each column j in X

    Assumes y and all columns of X are already residualized w.r.t covariates.
    Returns beta, se, t, p arrays for each column.
    """
    Sxx = np.sum(X * X, axis=0)
    valid = Sxx > 0

    beta = np.full(X.shape[1], np.nan, dtype=float)
    se   = np.full(X.shape[1], np.nan, dtype=float)
    tval = np.full(X.shape[1], np.nan, dtype=float)
    pval = np.full(X.shape[1], np.nan, dtype=float)

    X_valid = X[:, valid]
    XTy = X_valid.T @ y
    beta_valid = XTy / Sxx[valid]

    y_hat = X_valid @ beta_valid
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    mse = rss / df_resid

    se_valid = np.sqrt(mse / Sxx[valid])
    t_valid = beta_valid / se_valid
    p_valid = 2.0 * stats.t.sf(np.abs(t_valid), df=df_resid)

    beta[valid] = beta_valid
    se[valid] = se_valid
    tval[valid] = t_valid
    pval[valid] = p_valid

    return beta, se, tval, pval


def load_eqtl_genes(eqtl_path: str, p_cutoff: float) -> List[str]:
    """
    Load eQTL results and return unique genes passing p_cutoff.
    Expects some gene column and some p-value column; tries to standardize names.
    """
    eqtl = pd.read_csv(eqtl_path, sep="\t")

    colmap = {}
    for c in eqtl.columns:
        cl = c.lower()
        if cl in ["gene", "gene_id"]:
            colmap[c] = "gene"
        if cl in ["p-value", "pvalue", "p", "pval"]:
            colmap[c] = "p"
    eqtl = eqtl.rename(columns=colmap)

    if "gene" not in eqtl.columns or "p" not in eqtl.columns:
        raise ValueError(f"Could not find 'gene' and 'p' columns in eQTL file. "
                         f"Columns found: {eqtl.columns.tolist()}")

    eqtl["gene_upper"] = eqtl["gene"].astype(str).str.strip().str.upper()
    eqtl["p"] = pd.to_numeric(eqtl["p"], errors="coerce")

    eqtl_sig = eqtl.loc[eqtl["p"] <= p_cutoff, "gene_upper"].dropna().unique().tolist()
    eqtl_genes = sorted(eqtl_sig)
    return eqtl_genes


def pick_traits(morph: pd.DataFrame, traits_csv: str | None, use_traits_csv: bool) -> List[str]:
    if use_traits_csv:
        if traits_csv is None:
            raise ValueError("--use_traits_csv set but --traits_csv not provided.")
        # supports 1-column csv/tsv with trait names
        df = pd.read_csv(traits_csv, header=None)
        requested = df.iloc[:, 0].astype(str).tolist()
        traits = [t for t in requested if t in morph.columns]
        if len(traits) == 0:
            raise ValueError("No requested traits were found in morph columns.")
        return traits

    # otherwise scan all numeric morphology columns
    traits = [c for c in morph.columns if c != "donor" and pd.api.types.is_numeric_dtype(morph[c])]
    if len(traits) == 0:
        raise ValueError("No numeric morphology traits found in morph file.")
    return traits


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run eQTL-restricted cmTWAS scan: morphology ~ expression.")
    p.add_argument("--morph_csv", required=True, help="Morphology table (rows=samples, cols=traits + donor).")
    p.add_argument("--expr_csv", required=True, help="Expression table (rows=samples, cols=genes + donor).")
    p.add_argument("--eqtl_tsv", required=True, help="eQTL results table (tab-separated).")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--eqtl_p_cutoff", type=float, default=0.05, help="eQTL p-value cutoff to define eQTL genes.")
    p.add_argument("--min_n", type=int, default=30, help="Minimum non-missing samples for a trait to be tested.")
    p.add_argument("--traits_csv", default=None, help="Optional file containing trait names (1 per line).")
    p.add_argument("--use_traits_csv", action="store_true", help="If set, scan only traits provided in --traits_csv.")
    p.add_argument("--covar_num", nargs="*", default=[], help="Numeric covariate column names in morph_csv.")
    p.add_argument("--covar_cat", nargs="*", default=[], help="Categorical covariate column names in morph_csv.")
    p.add_argument("--top_traits_for_hits", type=int, default=10, help="How many top traits (by min p) to include in top hits table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # save config for reproducibility
    config_path = os.path.join(args.out_dir, "cmTWAS_run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # load
    morph = pd.read_csv(args.morph_csv)
    expr = pd.read_csv(args.expr_csv)

    morph = normalize_donor_col(morph)
    expr = normalize_donor_col(expr)
    expr = upper_gene_cols(expr)

    # eQTL genes
    eqtl_genes = load_eqtl_genes(args.eqtl_tsv, args.eqtl_p_cutoff)
    expr_genes_available = set([c for c in expr.columns if c != "donor"])
    genes = [g for g in eqtl_genes if g in expr_genes_available]
    if len(genes) == 0:
        raise ValueError("No eQTL genes overlap with expression matrix columns after uppercasing.")

    # align donors
    common = sorted(set(morph["donor"]).intersection(set(expr["donor"])))
    if len(common) == 0:
        raise ValueError("No overlapping donors between morph and expression files.")
    morph = morph.set_index("donor").loc[common].copy()
    expr = expr.set_index("donor").loc[common, genes].copy()

    # traits
    target_traits = pick_traits(morph.reset_index(), args.traits_csv, args.use_traits_csv)

    # covariate matrix from morph (need donor column present)
    morph_reset = morph.reset_index()
    C_df = build_covariate_matrix(morph_reset, args.covar_num, args.covar_cat)
    C_df.index = morph.index
    C = C_df.values
    n, k = C.shape
    if n - k - 1 <= 5:
        raise ValueError(f"Too few residual degrees of freedom (n={n}, k={k}). Reduce covariates.")

    # residualize expression once
    X = expr.values.astype(float)
    X_resid = residualize_matrix(X, C)

    summary_rows = []

    for trait in target_traits:
        y_raw = pd.to_numeric(morph[trait], errors="coerce").values.astype(float)
        ok = ~np.isnan(y_raw)

        if int(ok.sum()) < args.min_n:
            continue

        y = y_raw[ok]
        C_sub = C[ok, :]
        X_sub = X_resid[ok, :]

        df_trait = int(ok.sum()) - C_sub.shape[1] - 1
        if df_trait <= 5:
            continue

        y_resid = residualize_matrix(y.reshape(-1, 1), C_sub).ravel()

        beta, se, tval, pval = assoc_scan_y_vs_manyX(y_resid, X_sub, df_trait)

        qval = np.full_like(pval, np.nan, dtype=float)
        valid_p = ~np.isnan(pval)
        if int(valid_p.sum()) > 0:
            qval[valid_p] = multipletests(pval[valid_p], method="fdr_bh")[1]

        out = pd.DataFrame(
            {
                "trait": trait,
                "gene": genes,
                "beta": beta,
                "se": se,
                "t": tval,
                "p": pval,
                "q": qval,
                "n": int(ok.sum()),
                "df": int(df_trait),
            }
        ).sort_values("p", na_position="last")

        out_path = os.path.join(args.out_dir, f"cmTWAS_{trait}.sumstats.tsv")
        out.to_csv(out_path, sep="\t", index=False)

        min_p = float(np.nanmin(pval)) if np.any(~np.isnan(pval)) else np.nan
        min_q = float(np.nanmin(qval)) if np.any(~np.isnan(qval)) else np.nan
        n_sig_q05 = int(np.nansum(qval < 0.05))

        summary_rows.append(
            {
                "trait": trait,
                "n_samples": int(ok.sum()),
                "n_genes_tested": int(valid_p.sum()),
                "min_p": min_p,
                "min_q": min_q,
                "n_sig_genes_q05": n_sig_q05,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("min_p")
    summary_path = os.path.join(args.out_dir, "cmTWAS_summary_per_trait.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)

    # plots + top hits table
    if len(summary) > 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        vals = -np.log10(summary["min_p"].clip(lower=1e-300).values)
        plt.hist(vals, bins=30)
        plt.xlabel("-log10(min p) across eQTL genes (per trait)")
        plt.ylabel("Number of morphology traits")
        plt.title("cmTWAS signal distribution across morphology traits")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_minp_per_trait.png"), dpi=300)
        plt.close()

        # top hits table
        top_hits_rows = []
        top_traits = summary["trait"].head(args.top_traits_for_hits).tolist()
        for trait in top_traits:
            fpath = os.path.join(args.out_dir, f"cmTWAS_{trait}.sumstats.tsv")
            df = pd.read_csv(fpath, sep="\t").dropna(subset=["p"]).head(5)
            for r in df.itertuples(index=False):
                top_hits_rows.append(
                    {
                        "trait": r.trait,
                        "gene": r.gene,
                        "beta": r.beta,
                        "p": r.p,
                        "q": r.q,
                        "n": r.n,
                    }
                )

        top_hits = pd.DataFrame(top_hits_rows)
        top_hits_path = os.path.join(args.out_dir, "cmTWAS_top_hits_table.tsv")
        top_hits.to_csv(top_hits_path, sep="\t", index=False)

    print(f"[DONE] Wrote results to: {args.out_dir}")
    print(f"       Config saved to: {config_path}")
    print(f"       Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
