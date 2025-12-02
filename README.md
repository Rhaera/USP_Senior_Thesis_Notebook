# Functional Connectomes and Lightweight Ensembles for ADHD Classification

This repository contains the code and analysis for my senior thesis in Statistics and Data Science at the University of São Paulo (USP).  
The project investigates whether **functional connectomes** derived from **resting-state fMRI (rs-fMRI)**, combined with **complex network metrics** and **parsimonious supervised models**, can discriminate individuals with **ADHD** from **controls** in a **small-sample** regime (≈30 subjects).

The pipeline builds subject-level **functional connectivity matrices** using the **AAL116** and **Schaefer200** atlases, derives **graph-theoretical features**, constructs **class-specific connectivity templates**, and trains three main models:

- **Logistic Regression with L2 penalty (Ridge)**
- **XGBoost with depth-1 trees (decision stumps)**
- A lightweight **TemplateMargin Model (TMM)** based on three similarity margins

These models are then combined via a **simple soft-voting ensemble** (mean of predicted probabilities).  
Evaluation is performed with **repeated stratified cross-validation (5×5)**, with careful **confound regression**, **atlas gating**, and **statistical comparison** using the **Friedman** and **Nemenyi** tests.

---

## 1. Project Overview

### 1.1. Scientific Goal

The main goal is to explore whether **functional connectomes** and **complex network features** can be used to extract **discriminative signatures of ADHD** under:

- **High-dimensional** feature space (thousands of edges)
- **Very small sample size** (n ≈ 30)
- Need for **interpretability** and **robust validation**

Key ideas:

1. Represent each subject’s brain as a **weighted graph**, where:
   - Nodes = cortical/subcortical regions of interest (ROIs)
   - Edges = Fisher-z–transformed correlations between ROI time series

2. Build **class-specific templates**:
   - One average connectome for **controls**
   - One average connectome for **ADHD**
   - Use them to define **margins of similarity** for each subject

3. Train **simple, interpretable models** instead of heavy deep learning, to avoid severe overfitting in the small-sample regime.

---

## 2. Data

### 2.1. Source Dataset

The code assumes preprocessed rs-fMRI data from the **Neuro Bureau ADHD-200 Preprocessed** repository, in particular the Peking subset.

- Main repository: ADHD-200 Preprocessed
- Example download link for the `.tar` archive used in the notebook (Peking test set):

  ```text
  https://nitrc.org/frs/downloadlink.php/3814
