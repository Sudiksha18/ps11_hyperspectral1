# EUCLIDEAN_TECHNOLOGIES — Hyperspectral Anomaly Detection

An end-to-end pipeline to detect anomalies in PRISMA hyperspectral imagery using an enhanced Adaptive Mahalanobis Distance approach with optional target detection via ACE (Adaptive Cosine/Coherence Estimator).

- Zero data-loss preprocessing and memory-safe chunked processing
- Robust background modeling and ZCA whitening for stable distances
- Ensemble, multi-scale thresholding over multiple k-values
- Rich visualizations, GeoTIFF export, and leaderboard-ready packaging

## Project structure

- `src/main.py` — Orchestrates the full pipeline and generates outputs
- `src/data_loader.py` — Loads PRISMA `.he5` cubes (SWIR/VNIR), normalization, RGB preview
- `src/mahalanobis.py` — Adaptive Mahalanobis/ACE scoring, whitening, thresholds
- `src/evaluation.py` — ROC/PR metrics, plots, per-k evaluation
- `src/visualization.py` — Overlays, heatmaps, multi-k plots, GeoTIFF export
- `src/targets.py` — Synthetic man‑made target signatures (asphalt, concrete, metal, plastic)
- `dataset/PRS_L2D_...he5` — Sample input cube (expected path used by `main.py`)

Outputs are written to `EUCLIDEAN_TECHNOLOGIES_Hyperspectral_Outputs/`:
- `2_AnomalyDetectionResults/` — Maps, overlays, heatmaps, GeoTIFF
- `3_AccuracyReport/` — Accuracy report (Excel/CSV)
- `4_ModelDocumentation/` — PDF/TXT model documentation
- `README.txt` — Submission readme

## How the anomaly detection works

1) Load and normalize hyperspectral cube
- Data: PRISMA `.he5` file. `data_loader.load_hyperspectral_data()` reads SWIR and/or VNIR cubes, handles NaN/Inf, optional normalization to [0,1].
- For preview, `create_rgb_preview()` uses PCA or band picks to build an RGB image.

2) Background statistics with stability enhancements
- Flatten cube to X ∈ R^{n_pixels × n_bands}.
- Estimate mean μ and covariance Σ over background samples (Ledoit–Wolf optional).
- Compute an optimal regularization λ via model-based scoring; build Σ_reg = Σ + λI and robust Σ_reg^{-1} (Cholesky with SVD fallback).
- Optionally compute ZCA whitening W for numerically stable distances.

3) Per-pixel scoring (two modes)
- Unsupervised anomaly: Mahalanobis distance D(x) = sqrt((x-μ)^T Σ^{-1} (x-μ)), or whitened L2 after ZCA.
- Targeted anomaly (default in `main.py`): ACE score against synthetic man‑made signatures S (asphalt/concrete/metal/plastic) from `targets.py`:
  ACE(x,s) = (x^T Σ^{-1} s)^2 / [(s^T Σ^{-1} s)(x^T Σ^{-1} x)], taking the max over targets per pixel.

4) Ensemble adaptive thresholding across k-values
- Compute robust statistics of the score image (median and MAD).
- For k ∈ [0.3,0.5,0.7,1.0,1.2,1.5,1.8,2.0], build masks via:
  - Median+MAD thresholds: T_k = median + k·MAD
  - Percentile thresholds (e.g., 95–99.99th)
- Combine method-specific masks per k via consensus; union across k yields the final anomaly mask.

5) Reporting and exports
- Visualizations: anomaly map on RGB, per‑k B/W grids, combined intensity and counts, ACE heatmap.
- GeoTIFF export of boolean mask for leaderboard submission.
- Metrics with synthetic ground truth for demo (ROC/PR, F1/accuracy per k).
- Accuracy report (Excel if pandas/openpyxl available, else CSV).
- Model documentation (PDF via reportlab, else TXT fallback).

## Installation

- Python 3.9+ recommended.
- Install dependencies:
  - Minimal: `pip install -r requirements.txt`
  - Optional (for richer reports/GeoTIFFs): `pip install pandas openpyxl reportlab rasterio`

## Run

- Ensure the dataset exists at `dataset/PRS_L2D_STD_20241205050514_20241205050518_0001.he5` (or edit `data_path` in `src/main.py`).
- Run the full pipeline:

```
python src/main.py
```

This will create `EUCLIDEAN_TECHNOLOGIES_Hyperspectral_Outputs/` with results, reports, and a leaderboard package under `leaderboard_submission/`.

## Software requirements (detailed)

- Python: 3.9 — 3.11 recommended (3.11 used in tests here).
- Core dependencies (required):
  - numpy
  - scipy
  - h5py
  - opencv-python (cv2)
  - matplotlib
  - scikit-learn (optional but used for Ledoit-Wolf covariance)

- Recommended for full feature set (install if available):
  - pandas, openpyxl  (Excel report)
  - reportlab          (PDF model report)
  - rasterio or GDAL   (georeferenced GeoTIFF export)
  - rasterio.windows   (optional for windowed IO)

- Install via pip (example):

```powershell
python -m pip install -r requirements.txt
# (optional extras)
python -m pip install pandas openpyxl reportlab rasterio scikit-learn
```

If you don't have rasterio/GDAL, the exporter falls back to a numpy-based TIFF writer (no georeference). If scikit-learn is missing, the pipeline uses a numeric fallback for covariance estimation.

## Outputs created by the pipeline

All outputs are written under `EUCLIDEAN_TECHNOLOGIES_Hyperspectral_Outputs/` and a small `leaderboard_submission/` folder at the repo root. Key artifacts produced by `src/main.py`:

- `2_AnomalyDetectionResults/`
  - `EUCLIDEAN_TECHNOLOGIES_PRISMA_AnomalyMap.tif` — Boolean GeoTIFF for leaderboard (0 normal, 1 anomaly)
  - `EUCLIDEAN_TECHNOLOGIES_PRISMA_AnomalyMap.png` — RGB visualization with anomaly overlay
  - `EUCLIDEAN_TECHNOLOGIES_PRISMA_Overlay.png` — RGB preview overlaid with anomalies
  - `EUCLIDEAN_TECHNOLOGIES_PRISMA_MultiK_BW.png` — grid of black/white per-k detections
  - `EUCLIDEAN_TECHNOLOGIES_PRISMA_Combined_BW.png` — combined multi-k B/W map

- `3_AccuracyReport/`
  - `evaluation_plots.png` — ROC/PR, threshold diagnostics
  - `EUCLIDEAN_TECHNOLOGIES_AccuracyReport.xlsx` or `.csv` — Excel or CSV accuracy report

- `4_ModelDocumentation/`
  - `EUCLIDEAN_TECHNOLOGIES_ModelReport.pdf` or `.txt` — model documentation

- `leaderboard_submission/`
  - `PRISMA_anomaly_detection_<timestamp>.tif` — packaged GeoTIFF(s) ready to upload
  - README and metadata text files included with the submission

- `output/`
  - `anomaly_mask_boolean.npy` — saved binary mask backup

These files are produced automatically by `src/main.py`. If you run the optional `src/fast_submission.py` helper it will create a compact submission package and human-readable metadata files.

## Hardware requirements (recommended)

The pipeline is designed to run on commodity servers and workstations. Recommended minimum hardware for reasonable runtime on PRISMA-size cubes (the sample cube is ~269M values):

- CPU: 4+ cores (x86_64) — more cores reduce processing time for chunked operations
- RAM: 32 GB recommended (16 GB minimum for smaller scenes). The pipeline uses chunked processing to avoid exhausting RAM, but more RAM speeds up covariance and whitening steps.
- Disk: 50 GB free to store intermediate outputs and visualizations (SSD recommended)
- GPU: Not required. All processing is CPU-based. If you add GPU-accelerated linear algebra (CuPy or RAPIDS) you may speed up heavy matrix operations, but this is optional and unsupported in the reference implementation.

Notes:
- For very large scenes or batch processing, use a machine with 64+ GB RAM and many cores (16+). The pipeline's chunking parameters (`chunk_size`) can be increased to trade memory for speed.
- On Windows, ensure PowerShell and Python are configured to access the dataset path; on Linux, adjust file permissions for GeoTIFF/GDAL if used.

## Switching detection modes

- Default: targeted detection via ACE (`detection_mode='target_ace'` with `get_manmade_signatures`).
- Unsupervised: change to `detection_mode='anomaly'` in `src/mahalanobis.py` initialization, which uses Mahalanobis/whitened L2.

## Notes

- If GeoTIFF backends (rasterio/GDAL) are unavailable, the exporter falls back to TIFF/PNG without georeferencing.
- Excel/PDF outputs have CSV/TXT fallbacks when optional deps are missing.
- Processing uses chunking to handle large images without exhausting RAM.