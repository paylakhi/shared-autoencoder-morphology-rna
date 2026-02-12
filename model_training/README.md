# Shared Latent Autoencoder for RNA–Morphology Integration

This repository contains the training implementation of the shared latent-space autoencoder described in:

**Cellular morphology emerges from polygenic, distributed transcriptional variation**  
Paylakhi et al., 2026

---

## Overview

This model learns a shared latent representation between:

- RNA-seq gene expression profiles
- High-dimensional cellular morphology features

The architecture consists of:

- RNA encoder
- Morphology encoder
- Shared latent space
- Modality-specific decoders

After training, morphology is predicted from RNA using:

RNA Encoder → Shared Latent Space → Morphology Decoder

This framework enables cross-modal prediction and analysis of distributed gene–morphology relationships.

---

## Datasets

The model was trained independently on:

- LUAD
- LINCS
- TOARF
- CDRP

Each dataset is trained separately using the same architecture.

---

## Model Architecture

- Fully connected encoders and decoders
- Shared latent dimension (default: 150)
- Mean Squared Error (MSE) reconstruction loss
- Gradient clipping for stability
- Early stopping based on validation loss

---

## Reproducibility

To ensure reproducibility:

- RNA and morphology splits use identical sample indices
- Random seed is fixed
- Training configuration is exported
- Model weights and logs are saved automatically

---
