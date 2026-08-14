# Multimodal Brain Tumor MRI Classification & Explainability (CAD)

An end-to-end Computer-Aided Diagnosis (CAD) pipeline benchmarking modern Deep Learning architectures (ConvNeXt vs. Swin Transformer) on Brain MRI multi-class detection.

## Project Structure
- `data/metadata/`: Metadata registries and exploratory manifestations.
- `notebooks/`: Modular experimental notebooks adhering to the 8-step AMIT data engineering rubric.
- `src/`: Core preprocessing, model architectures, and training scripts.

## Step 1: Data Exploration & Validation
- **Dataset Size:** 7200 MRI scans across 4 classes (`glioma`, `meningioma`, `pituitary`, `notumor`).
- **Data Integrity:** 100% readable files, verified channel shapes, and dimension distributions cataloged in `data/metadata/metadata_raw.csv`.

## Step 2: Cryptographic Deduplication & Data Integrity Audit
- **Methodology:** Implemented 128-bit MD5 cryptographic checksums across raw pixel buffers to detect duplicate scans across subdirectories.
- **Duplicates Identified:** Detected and dropped **187 redundant scans (2.60%)**, filtering 340 duplicate instances down to a clean cohort of **7,013 unique MRI images**.
- **Leakage & Conflict Audit:** Confirmed **0 cross-split leakage cases** (train $\leftrightarrow$ test) and **0 diagnostic label conflicts**.
- **Artifact Generated:** Exported cleaned manifest to `data/metadata/metadata_deduplicated.csv`.

## Step 3: Missing Value Audit & Imputation Benchmark
- **Audit:** Confirmed $0.0\%$ tabular missing values and $100\%$ pixel tensor integrity (zero unreadable or blank slices across all $7,013$ scans).
- **Controlled Simulation:** Benchmarked missing data handling on a simulated $3.44\%$ Missing-Completely-At-Random (MCAR) feature cohort.
- **Technique Comparison:**
  - *Listwise Deletion:* Resulted in a $19.11\%$ reduction in usable samples ($5,673$ retained).
  - *Univariate (Mean/Median):* Higher reconstruction bias ($\text{MAE} = 6.9032$ / $4.9079$).
  - *Multivariate (KNN, $k=5$):* Lowest reconstruction error (**$\text{MAE} = 1.2029$**), preserving complex feature correlations.
  - *Sequential (Forward/Backward Fill):* Effectively retained predictive stability ($76.24\%$ CV accuracy) for spatial slice series.
- **Artifact:** Exported verified dataset to `data/metadata/metadata_clean_step3.csv`.

| Imputation Strategy | CV Accuracy (%) | CV Macro F1 (%) | Reconstruction MAE |
| :--- | :--- | :--- | :--- |
| **Complete (Ground Truth)** | $76.20\% \pm 1.02\%$ | $76.33\% \pm 1.01\%$ | $0.0000$ |
| **Listwise Deletion (<5%)** | $75.48\% \pm 0.71\%$ | $75.63\% \pm 0.72\%$ | N/A (Dropped) |
| **Mean Imputation** | $75.06\% \pm 1.17\%$ | $75.12\% \pm 1.19\%$ | $6.9032$ |
| **Median Imputation** | $75.23\% \pm 1.09\%$ | $75.28\% \pm 1.10\%$ | $4.9079$ |
| **Mode Imputation** | $75.63\% \pm 1.10\%$ | $75.67\% \pm 1.10\%$ | $5.7928$ |
| **KNN Imputation ($k=5$)** | $75.45\% \pm 0.76\%$ | $75.53\% \pm 0.76\%$ | **$1.2029$** |
| **Forward Fill** | **$76.24\% \pm 0.89\%$** | **$76.31\% \pm 0.89\%$** | $4.4624$ |
| **Backward Fill** | $76.10\% \pm 0.92\%$ | $76.17\% \pm 0.94\%$ | $4.0302$ |

## Step 4: Outlier Detection, Handling & ROI Extraction
- **Detection Framework:** Benchmarked anomalies across Z-score ($|Z| > 3$), IQR ($1.5 \times \text{IQR}$), and Isolation Forest (3% contamination) on intensity and dimension metrics.
- **Handling Strategies:**
  - *Transformations:* Applied Log and Square-Root transformations to stabilize right-skewed distributions.
  - *Capping:* Winsorized continuous features at the 1st and 99th percentiles.
  - *Filtering:* Dropped **358 multi-method consensus outliers (5.10%)** to eliminate severely corrupted/distorted scans.
- **Morphological ROI Extraction:** Implemented OpenCV extreme contour cropping to eliminate dead background margins and standardize regions of interest to $224 \times 224$.
- **Artifact:** Exported curated cohort of **6,655 verified scans** to `data/metadata/metadata_clean_step4.csv`.

| Stage / Metric | Step 3 Clean | Step 4 Processed |
| :--- | :--- | :--- |
| **Total Scans** | 7,013 | 6,655 |
| **Outliers Removed** | 0 | 358 (5.10%) |
| **ROI Standardization** | Variable | 224 × 224 (Contour Cropped) |

## Step 5: Exploratory Data Visualization & Quality Audit
- **Visual Suite:** Generated 7 publication-grade exploratory figures covering univariate KDE densities, correlation heatmaps, bivariate feature interactions, multivariate pairplots, and anatomical MRI slice grids.
- **Class Stratification Audit:** Formally verified the 4-class distribution across all 6,655 curated scans:
  - `No Tumor`: 1,858 scans (27.92%)
  - `Pituitary`: 1,680 scans (25.24%)
  - `Glioma`: 1,560 scans (23.44%)
  - `Meningioma`: 1,557 scans (23.40%)
- **Feature Covariance:** Quantified dimensional coupling ($r = 0.94$ for height vs. file size; $r = 0.44$ for mean intensity vs. standard deviation), proving that first-order pixel statistics have non-linear class overlap that requires deep neural representation learning.
- **Pipeline Quality Matrix:** Validated data integrity across all stages (7,200 raw $\rightarrow$ 7,013 deduplicated $\rightarrow$ 6,655 clean scans).

| Class Label | Scan Count | Percentage (%) | Distribution Status |
| :--- | :--- | :--- | :--- |
| **No Tumor (`notumor`)** | 1,858 | 27.92% | Balanced |
| **Pituitary (`pituitary`)** | 1,680 | 25.24% | Balanced |
| **Glioma (`glioma`)** | 1,560 | 23.44% | Balanced |
| **Meningioma (`meningioma`)** | 1,557 | 23.40% | Balanced |
| **Total Cohort** | **6,655** | **100.00%** | Verified |