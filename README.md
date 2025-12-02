```
# Modeling of Functional Connectomes with Complex Networks in Reduced Sampling Scenarios for Predicting Attention Deficit Hyperactivity Disorder

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
````

* Phenotypic file (TSV with subject-level labels):

  ```text
  https://nitrc.org/frs/download.php/9024/adhd200_preprocessed_phenotypics.tsv
  ```

> ⚠️ **Important:**
> The repository **does not** redistribute the original fMRI data.
> You must download the dataset yourself from the original source (NITRC / Neuro Bureau), respecting all licenses and usage terms.

### 2.2. Derived Files

The notebook exports the following CSV files (stored under `RHARA_TCC/features`):

* `aal_connectomes.csv` – functional connectomes using the **AAL116** atlas
* `schaefer200_connectomes.csv` – functional connectomes using **Schaefer200 (Yeo7, 200 parcels)**
* `combined_connectomes.csv` – merged AAL + Schaefer edges per subject
* `fd_summary.csv` – summary of framewise displacement (FD) and scrubbing
* `labels.csv` – subject IDs, diagnosis (ADHD vs control) and site information

These are the inputs used in the **main modeling script**.

---

## 3. Environment and Dependencies

The code was originally developed and executed in **Google Colab**, using **Python 3.x**.

### 3.1. Core Dependencies

Main libraries used:

* `numpy`
* `pandas`
* `matplotlib`
* `scipy`
* `networkx`
* `scikit-learn`
* `xgboost`
* `nilearn`
* `scikit-posthocs`
* `SimpleITK`
* `h5py` (for robust motion file parsing)
* `gzip`, `json`, `tempfile`, `pathlib`, etc. (standard library)

In Colab, the non-native libraries were installed with:

```bash
pip install nilearn
pip install scikit-posthocs
pip install xgboost
pip install SimpleITK
```

If running locally, you can create a virtual environment and install the same dependencies via `pip` or `conda` as needed.

---

## 4. Repository Structure

A typical layout for this project might look like:

```text
.
├── Senior_Thesis_Notebook.ipynb   # Main Jupyter/Colab notebook (fully commented in English)
├── tcc_main_script.py             # (Optional) Exported main script for the modeling part
├── README.md                      # This file
└── results/                       # Output metrics, plots and tables (created at runtime)
    ├── summary_metrics.csv
    ├── summary_metrics_with_ci95.csv
    ├── r_auc_list_*.npy
    ├── r_acc_list_*.npy
    ├── metricas_combinadas_por_dobra.csv
    ├── nemenyi_heatmap_metric_composta.png
    ├── roc_ensemble_oof.png
    ├── hist_scores_ensemble.png
    └── ensemble_threshold_global.json
```

On Google Drive (as expected by the notebook), the base folder is:

```text
/content/drive/MyDrive/RHARA_TCC/
```

with subfolders like:

* `peking_test_output_nifty/` – extracted fMRI data and intermediates
* `features/` – exported connectome CSVs and labels
* `results/` – all downstream model evaluation outputs

---

## 5. Pipeline Summary

### 5.1. Step 1 – Mount Drive and Extract Raw Data

The notebook:

1. Mounts Google Drive in Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Sets a target folder (e.g., `/content/drive/MyDrive/RHARA_TCC`).

3. Locates and extracts the `.tar` archive containing the preprocessed Peking dataset using `tar`.

### 5.2. Step 2 – Subject Discovery and Motion Handling

Utilities:

* **`discover_subjects()`**
  Scans the fMRI directory for files like:

  * `func_*_session_1_run1*.nii(.gz)`
  * `fmri_*_session_1_run1*.nii(.gz)`

  and tries to find matching motion parameter files (MAT, TXT, PAR, TSV, CSV, gzipped).

* **`load_motion_params_auto()`**
  Robust loader that:

  * Handles MATLAB `.mat` (v5/v7, v7.3/HDF5)
  * Handles plain-text ASCII motion files
  * Normalizes to a `(T, 6)` matrix: `[Tx, Ty, Tz, Rx, Ry, Rz]`

* **`compute_fd_tr()`**
  Computes **framewise displacement (FD) per TR** (Power et al.) from translation and rotation parameters.

* **SimpleITK fallback**
  If motion parameters are missing or degenerate, the code estimates 6 DOF motion directly from the 4D NIfTI using **SimpleITK**.

### 5.3. Step 3 – Scrubbing and Time Series Extraction

For each subject:

1. Compute FD per TR and apply **scrubbing**:

   * Keep only TRs with FD ≤ 0.2 mm
   * Require at least **100 volumes after scrubbing** (`MIN_TR_AFTER = 100`)

2. Build an **EPI mask** with `nilearn.masking.compute_epi_mask`.

3. Extract ROI time series using `NiftiLabelsMasker` for:

   * **AAL116 atlas (SPM12 version)**
   * **Schaefer200 (Yeo7, 2mm)**

   Parameters:

   * `standardize = "zscore_sample"` (z-score per ROI across time)
   * `detrend = True`
   * `smoothing_fwhm = None`

### 5.4. Step 4 – Connectivity and Fisher-z Transform

For each atlas and subject:

1. Compute correlation matrix `C` over ROIs with sufficient variance.

2. Apply Fisher-z transform:

   [
   Z = \operatorname{arctanh}(C)
   ]

3. Extract the upper-triangular part of `Z` as a **feature vector** (one per subject and per atlas).

All subject-level vectors are saved as:

* `aal_connectomes.csv`
* `schaefer200_connectomes.csv`
* `combined_connectomes.csv` (merge by `subject_id`)

Additionally, motion / scrubbing summaries are saved in `fd_summary.csv`.

### 5.5. Step 5 – Labels and Phenotypics

The notebook reads:

* `adhd200_preprocessed_phenotypics.tsv`
* Detects the ID and diagnosis columns
* Normalizes subject IDs (e.g., `X_1038415 → 1038415`)
* Maps numeric diagnosis to labels:

  * `1 → TDC`
  * `2 → ADHD`

and writes:

* `labels.csv` (with `subject_id`, `diagnosis`, and `site`)

### 5.6. Step 6 – Feature Construction for Machine Learning

Using `combined_connectomes.csv`, `labels.csv` and `fd_summary.csv`, the main script:

1. Builds **subject-level matrices** from edge vectors via `vec_to_mat`.

2. For each **outer CV fold**, it:

   * Regresses out **mean FD** as a confound from edge features
   * Builds AAL and Schaefer matrices for **all subjects in that fold**

3. Within each fold, it constructs:

   * **Class-specific templates** (control vs ADHD) per atlas using Fisher-z averaging
   * **Graph metrics** (density, strength, clustering, transitivity, assortativity, global efficiency, etc.)
   * **Template margins**:

     * Cosine similarity margin
     * Pearson correlation margin
     * Frobenius-distance margin
   * **Top-M edges** with largest absolute difference between templates

4. The features are generated for three spaces:

   * **AAL only**
   * **Schaefer only**
   * **Both (concatenated)**

Additionally, “TemplateMargin-only” versions keep **only the three margin features**.

### 5.7. Step 7 – Models and Atlas Gating

Three families of models are used:

1. **Logistic Regression (Ridge)**

   * Pipeline:

     * `RobustScaler`
     * `SelectKBest(mutual_info_classif, k ∈ {30,45,60})`
     * `LogisticRegression(penalty="l2", class_weight="balanced")`

2. **XGBoost (stumps)**

   * `XGBClassifier` with:

     * `max_depth = 1`
     * `n_estimators = 300`
     * `learning_rate = 0.06`
     * `scale_pos_weight` calibrated from class imbalance
   * Feature selection with `SelectKBest` as above

3. **TemplateMargin Model (TMM)**

   * Logistic regression over **only three features**:

     * `margin_cos`
     * `margin_corr`
     * `margin_frob`

#### Atlas Gating

For each outer fold and each model type (`ridge`, `xgb`, `tm`):

* A **StratifiedKFold (3-fold)** inner CV is used with `HalvingGridSearchCV`
* It selects which feature space performs best in terms of **AUC**:

  * AAL vs Schaefer vs Both
* The best space + hyperparameters are then used to fit on the full training set of that outer fold.

### 5.8. Step 8 – Ensemble and Evaluation

The ensemble is a **simple soft-voting average**:

[
\hat{p}_{\mathrm{ens}}(x) =
\frac{1}{3}\bigl[p_R(x) + p_X(x) + p_T(x)\bigr]
]

where:

* ( p_R(x) ) = Ridge probability
* ( p_X(x) ) = XGBoost probability
* ( p_T(x) ) = TemplateMargin probability

Two types of thresholds are used:

* Fixed: ( \tau = 0.5 )
* Optimal: global Youden index threshold ( \tau_{\mathrm{Youden}} ), computed from **out-of-fold** predictions.

Metrics:

* **Per-fold** AUC and accuracy for each model

* **Global out-of-fold AUC** for the ensemble

* **Confidence intervals (95%)** for AUC and ACC using the t-distribution

* **Friedman test** over a composite metric

  [
  M = 0.7 \cdot \text{AUC} + 0.3 \cdot \text{ACC}
  ]

* **Nemenyi post-hoc** test and heatmap of pairwise p-values

---

## 6. How to Run (Colab Workflow)

1. Upload the notebook (`Senior_Thesis_Notebook.ipynb`) to Google Colab.

2. Mount Google Drive.

3. Download and place the `.tar` dataset and the phenotypic TSV in your `RHARA_TCC` folder.

4. Install the extra dependencies:

   ```bash
   pip install nilearn
   pip install scikit-posthocs
   pip install xgboost
   pip install SimpleITK
   ```

5. Run the notebook cells in order:

   * Data extraction and scrubbing
   * Connectome construction
   * Label processing
   * Main modeling loop (5×5 repeated CV)
   * Statistical tests (Friedman + Nemenyi)
   * Confidence interval computation

All results will be saved under:

```text
/content/drive/MyDrive/RHARA_TCC/results
```

---

## 7. Citation and References

If you use this code or ideas in your own work, please cite:

* The original ADHD-200 Preprocessed repository:
  Bellec et al. (2017), *The Neuro Bureau ADHD-200 Preprocessed Repository*, NeuroImage, 144 (Pt B), 275–286.
* Key complex network references:

  * Bullmore & Sporns (2009), *Complex brain networks: graph theoretical analysis of structural and functional systems*, Nat Rev Neurosci.
  * Rubinov & Sporns (2010), *Complex network measures of brain connectivity: uses and interpretations*, NeuroImage.
  * Basso & Sporns (2019), *Network science of the human brain: foundations and applications*, Nat Rev Neurosci.
* Cross-validation and small-sample issues:

  * Varoquaux (2018), *Cross-validation failure: Small sample sizes lead to large error bars*, NeuroImage.
* Statistical comparison of classifiers:

  * Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR.

You may also cite the senior thesis itself (add your final bibliographic entry here once it is officially archived).

---

## 8. License

```text
MIT License – see LICENSE file for details.
```

---

## 9. Contact

If you have questions about the code, methodology, or want to discuss potential collaborations, feel free to reach out via:

* GitHub issues in this repository
* Email via rauldantas@usp.br

---

**Note:**
All comments in the notebook have been translated to English (`en-US`) to facilitate review by graduate admissions committees and international collaborators, while the code itself remains unchanged.

```

---
```
