# Generalization to Individual-Level Donor Data (No Retraining)

This folder reproduces the manuscript evaluation where the LUAD-trained shared autoencoder
is applied to donor-level RNA-seq + morphology data without retraining.

## Summary

1. Load LUAD-trained checkpoint from `model_training/`
2. Harmonize donor RNA-seq to the L1000 gene set (intersection)
3. Harmonize morphology features to shared features across datasets
4. Evaluate two schemes:
   - Global evaluation: predict morphology for all donors and compute R² per feature
   - Individual-level evaluation: compute per-donor prediction error per feature and aggregate across donors

## Run

```bash
python apply_to_donor_data.py \
  --ckpt /path/to/LUAD_shared_autoencoder_ckpt.pt \
  --donor_rna_csv /path/to/donor_rnaseq.csv \
  --donor_morph_csv /path/to/donor_morph.csv \
  --l1000_gene_list /path/to/l1000_genes.txt \
  --shared_morph_list /path/to/shared_morph_features.txt \
  --out_dir /path/to/output
