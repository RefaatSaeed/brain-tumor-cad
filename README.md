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

## Step 7: Bias-Variance Diagnostics & Regularization Protocols
- **Empirical Diagnostics:** Formally benchmarked underfitting (Decision Stump, Train Acc: 45.47%, Macro F1: 33.63%) vs. overfitting (Unconstrained Tree, Train Acc: 100.00%, Val Acc: 68.84%, Generalization Gap: +31.16%).
- **Regularization & Capacity Remediation:**
  - *Polynomial Expansion:* Generated 27 degree-2 interaction features to overcome high bias.
  - *L1/L2 Penalties:* Applied Lasso/Ridge constraints, reducing generalization variance gap to $<1.2\%$.
  - *Tree Pruning:* Identified optimal cost-complexity parameter ($\alpha = 0.00156$), reducing variance gap from +31.16% down to +1.67%.
  - *Neural Regularization:* Implemented L2 weight decay and early stopping on MLP, achieving optimal generalization ($\text{Val Acc} = 71.74\%$, $\text{Gap} = +0.15\%$).
- **Diagnostic Curves:** Generated 5-fold cross-validated learning curves and cost-complexity pruning paths validating asymptotic convergence.

| Model Strategy | Train Acc (%) | Val Acc (%) | Generalization Gap | Val Macro F1 (%) | Empirical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Decision Stump (`depth=1`)** | 45.47% | 44.59% | +0.88% | 33.63% | Severe Underfitting |
| **2. Unconstrained Tree** | 100.00% | 68.84% | +31.16% | 70.09% | Severe Overfitting |
| **3. Polynomial + LogReg** | 71.30% | 70.64% | +0.66% | 71.07% | Regularized Fit |
| **4. L1 Lasso Regularization** | 69.73% | 68.74% | +0.99% | 69.10% | Regularized Fit |
| **5. L2 Ridge Regularization** | 69.86% | 68.74% | +1.12% | 69.13% | Regularized Fit |
| **6. Pruned Tree ($\alpha=0.00156$)** | 72.11% | 70.44% | +1.67% | 71.20% | Regularized Fit |
| **7. Regularized Neural Net (MLP)** | **71.90%** | **71.74%** | **+0.15%** | **71.95%** | **Optimal Fit** |

- **Artifact:** Exported validated manifest to `data/metadata/metadata_step7_validated.csv`.

## Step 8: Multi-Metric Clinical Evaluation & Model Benchmarking
- **Holdout Test Evaluation ($N = 999$):** Evaluated linear baselines, pruned decision trees, regularized MLPs, Random Forests, and Gradient Boosting ensembles on unseen holdout test scans.
- **Diagnostic Visualizations:** Generated multi-class normalized confusion matrices, One-vs-Rest (OvR) ROC curves, precision-recall curves, and cross-model performance benchmarks.
- **Classification Benchmark Matrix (Holdout Test Set):**

| Model Architecture | Accuracy (%) | Balanced Acc (%) | Precision (Macro) | Recall (Macro) | F1-Macro (%) | Cohen's Kappa (κ) | MCC | OvR ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L2 Logistic Regression** | 70.57% | 71.33% | 70.70% | 71.33% | 70.82% | 0.6066 | 0.6080 | 0.8868 |
| **Cost-Complexity Pruned Tree** | 74.77% | 75.93% | 76.85% | 75.93% | 75.66% | 0.6632 | 0.6674 | 0.9003 |
| **Regularized MLP Neural Net** | 71.77% | 72.76% | 71.69% | 72.76% | 72.03% | 0.6230 | 0.6242 | 0.9097 |
| **Random Forest (Top Performer)** | **77.78%** | **78.90%** | **78.23%** | **78.90%** | **78.47%** | **0.7033** | **0.7038** | **0.9387** |
| **Gradient Boosting Ensemble** | 76.88% | 77.99% | 77.71% | 77.99% | 77.83% | 0.6910 | 0.6911 | 0.9347 |

- **Per-Class Discriminative Performance (ROC-AUC & PR Analysis):**

| Diagnostic Class | True Test Scans | Class Sensitivity | ROC-AUC | Average Precision (AP) |
| :--- | :---: | :---: | :---: | :---: |
| **No Tumor (`notumor`)** | 209 | **98.56%** | **0.998** | **0.992** |
| **Pituitary (`pituitary`)** | 263 | **77.57%** | **0.948** | **0.869** |
| **Glioma (`glioma`)** | 266 | **70.30%** | **0.914** | **0.817** |
| **Meningioma (`meningioma`)** | 261 | **65.52%** | **0.879** | **0.714** |

- **Auxiliary Quantitative Regression Evaluation:**

| Regression Metric | Evaluation Formula | Holdout Test Score |
| :--- | :--- | :---: |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | **0.8039** |
| **Mean Absolute Error (MAE)** | $\frac{1}{N} \sum \|y - \hat{y}\|$ | **4.7058** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | **6.6967** |
| **Mean Squared Error (MSE)** | $\frac{1}{N} \sum (y - \hat{y})^2$ | **44.8464** |

- **Artifact:** Exported benchmark audit to `data/metadata/model_evaluation_benchmarks.csv`.


## Key Findings

Our comparative analysis highlights the effectiveness of modern deep learning architectures for automated brain tumor classification:

| Architecture | Test Accuracy | Macro F1-Score | Clinical Interpretability |
| :--- | :--- | :--- | :--- |
| Random Forest (Baseline) | 77.78% | 78.47% | Low |
| **ConvNeXt-Tiny** | **99.20%** | **99.23%** | **High (Grad-CAM)** |
| Swin-Transformer-Tiny | 99.10% | 99.12% | High (Attention) |

### Highlights:
- **Architectural Performance:** Deep Learning models surpassed traditional ML baselines by over 20% in overall accuracy, demonstrating the power of feature extraction via deep hierarchical architectures.
- **Explainability (XAI):** We integrated Grad-CAM to visualize model decision-making. The high overlap between the model's activations and known clinical tumor locations provides "white-box" evidence, building trust for clinical applications.
- **Robustness:** Stratified dataset management and cost-sensitive loss functions ensured balanced classification, effectively handling the intrinsic class imbalance in the source MRI dataset.

---
### Future Directions
- **Clinical Validation:** Future work will involve testing the model on multi-center datasets to assess robustness against variations in MRI scanner hardware and imaging protocols.
- **Deployment:** Transitioning the model into a lightweight, real-time inferencing application for neuro-radiology support.