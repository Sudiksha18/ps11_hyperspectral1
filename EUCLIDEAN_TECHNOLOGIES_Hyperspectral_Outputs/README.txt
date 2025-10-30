EUCLIDEAN_TECHNOLOGIES - Hyperspectral Anomaly Detection Submission
================================================================

SUBMISSION OVERVIEW
Team: EUCLIDEAN_TECHNOLOGIES
Date: 2025-10-13
Algorithm: Enhanced Adaptive Mahalanobis Distance Detection
Version: Enhanced_v2.0

DIRECTORY STRUCTURE
├── 1_HashValue/
│   └── EUCLIDEAN_TECHNOLOGIES_ModelHash.txt
├── 2_AnomalyDetectionResults/
│   ├── EUCLIDEAN_TECHNOLOGIES_PRISMA_AnomalyMap.tif    (GeoTIFF for leaderboard)
│   ├── EUCLIDEAN_TECHNOLOGIES_PRISMA_AnomalyMap.png    (Visualization)
│   └── EUCLIDEAN_TECHNOLOGIES_PRISMA_Overlay.png       (RGB overlay)
├── 3_AccuracyReport/
│   └── EUCLIDEAN_TECHNOLOGIES_AccuracyReport.xlsx      (Performance metrics)
├── 4_ModelDocumentation/
│   └── EUCLIDEAN_TECHNOLOGIES_ModelReport.pdf          (Algorithm details)
└── README.txt (this file)

🎯 PERFORMANCE SUMMARY
Overall Accuracy: 0.887
Best F1-Score: 0.055
ROC AUC: 0.500
PR AUC: 0.050
Total Anomalies: 10397 (5.00%)

🔧 TECHNICAL DETAILS
- Algorithm: Adaptive Mahalanobis Distance with ZCA Whitening
- Features: Zero data loss, ensemble thresholding, robust covariance
- Input: PRISMA Hyperspectral data (1280 bands)
- Output: Boolean anomaly map (0=normal, 1=anomaly)
- Processing: Multi-scale detection with enhanced regularization

📧 CONTACT
Team: EUCLIDEAN_TECHNOLOGIES
Email: team@euclidean-tech.com
Submission: October 13, 2025

✅ FILES READY FOR LEADERBOARD UPLOAD
Main file: 2_AnomalyDetectionResults/EUCLIDEAN_TECHNOLOGIES_PRISMA_AnomalyMap.tif
Format: GeoTIFF, Boolean values (0,1)
