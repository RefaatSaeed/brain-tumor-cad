# Multimodal Brain Tumor MRI Classification & Explainability (CAD)

An end-to-end Computer-Aided Diagnosis (CAD) pipeline benchmarking modern Deep Learning architectures (ConvNeXt vs. Swin Transformer) on Brain MRI multi-class detection.

## Project Structure
- `data/metadata/`: Metadata registries and exploratory manifestations.
- `notebooks/`: Modular experimental notebooks adhering to the 8-step AMIT data engineering rubric.
- `src/`: Core preprocessing, model architectures, and training scripts.

## Step 1: Data Exploration & Validation
- **Dataset Size:** 7200 MRI scans across 4 classes (`glioma`, `meningioma`, `pituitary`, `notumor`).
- **Data Integrity:** 100% readable files, verified channel shapes, and dimension distributions cataloged in `data/metadata/metadata_raw.csv`.