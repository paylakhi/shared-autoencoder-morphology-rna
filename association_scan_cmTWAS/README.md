# eQTL-Restricted Expression–Morphology Association Scan (TWAS-like)

This folder contains the implementation of the TWAS-like analysis described in:

**Cellular morphology emerges from polygenic, distributed transcriptional variation**  
Paylakhi et al., 2026

## Overview

For each morphology feature (treated as a phenotype), we test the linear association:

    morphology_trait ~ gene_expression (+ optional covariates)

The scan is restricted to genes identified as significant cis-eQTL genes in an external eQTL dataset. This reduces the multiple-testing burden and focuses the analysis on genes with regulatory evidence.
