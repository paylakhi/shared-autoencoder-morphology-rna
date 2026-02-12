# Data Preprocessing

This folder contains scripts for preprocessing RNA-seq gene expression and cellular morphology data prior to model training.

Main steps include:
- Normalization of gene expression data using counts per million (CPM)
- Standardization of gene expression features
- Scaling of morphology features to ensure comparability across datasets
- Outlier removal to reduce the influence of extreme values
- Alignment and matching of samples across modalities
- Implementation of all preprocessing steps in Python to ensure reproducibility
