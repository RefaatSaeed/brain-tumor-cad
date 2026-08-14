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