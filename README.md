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
- **Visual Suite:** Constructed 7 exploratory figures covering class balance, univariate KDE distributions, correlation heatmaps, bivariate feature interaction planes, multivariate pairplots, and slice mosaics.
- **Class Stratification Audit:** Formally verified the 4-class distribution across all 6,655 curated scans:
  - `Glioma`: 1,774 scans (26.66%)
  - `Pituitary`: 1,748 scans (26.27%)
  - `Meningioma`: 1,735 scans (26.07%)
  - `No Tumor`: 1,398 scans (21.01%)
- **Covariance & Feature Dependencies:** Quantified structural and statistical couplings ($r = +0.99$ for width vs. height; $r = +0.71$ for mean intensity vs. standard deviation).
- **Data Quality Matrix:** Confirmed the structured pipeline transition across ingestion stages:

| Processing Stage | Valid Scans | Missing Attributes | Cryptographic Duplicates | Severe Outliers Filtered |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Ingested** | 7,200 | 0 | 187 (2.60%) | 358 (5.10%) |
| **Step 3 (Cleaned Manifest)** | 7,013 | 0 | 0 | 358 |
| **Step 4 (Processed Cohort)** | **6,655** | **0** | **0** | **0** |

| Class Label | Scan Count | Percentage (%) | Distribution Balance |
| :--- | :---: | :---: | :---: |
| **Glioma (`glioma`)** | 1,774 | 26.66% | Balanced |
| **Pituitary (`pituitary`)** | 1,748 | 26.27% | Balanced |
| **Meningioma (`meningioma`)** | 1,735 | 26.07% | Balanced |
| **No Tumor (`notumor`)** | 1,398 | 21.01% | Balanced |
| **Total Cohort** | **6,655** | **100.00%** | Verified |

## Step 6: Class Imbalance Evaluation, Resampling & Stratified Partitioning
- **Imbalance Audit:** Confirmed a majority-to-minority ratio of **1.27:1** (`glioma`: 26.66% vs. `notumor`: 21.01%), satisfying standard multi-class balance conditions (<70/30 rule).
- **Cost-Sensitive Class Weights:** Computed balanced loss weights (`glioma`: 0.9379, `meningioma`: 0.9589, `pituitary`: 0.9518, `notumor`: 1.1901) for deep learning cross-entropy optimization.
- **Resampling Benchmarks:** Evaluated Random Undersampling (5,592 samples), SMOTE (7,096 samples), and ADASYN (7,011 samples) on tabular feature distributions.
- **Stratified Partitioning (70/15/15):** Created leak-free train, validation, and test subsets preserving class proportions across partitions:

| Partition Split | Ratio | Glioma | Meningioma | Pituitary | No Tumor | Total Scans |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 70.0% | 1,242 | 1,214 | 1,223 | 979 | **4,658** |
| **Validation Set** | 15.0% | 266 | 260 | 262 | 210 | **998** |
| **Test Set** | 15.0% | 266 | 261 | 263 | 209 | **999** |
| **Total Cohort** | **100.0%** | **1,774** | **1,735** | **1,748** | **1,398** | **6,655** |

- **Artifact:** Exported final partitioned manifest to `data/metadata/metadata_stratified_step6.csv`.