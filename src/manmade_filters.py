"""
Man-made anomaly filtering utilities for hyperspectral cubes.
Applies semantic masks (vegetation, water) and built-up index to constrain
anomalies to anthropogenic targets.
"""

import numpy as np
import cv2
from typing import Tuple

EPS = 1e-6


def _band_index_for_wavelength(n_bands: int, wavelength_nm: float) -> int:
    """
    Approximate band index for a target wavelength assuming the cube spans
    ~400-2500 nm uniformly (VNIR+SWIR). This is a heuristic when band centers
    are not available.
    """
    wl_min, wl_max = 400.0, 2500.0
    frac = np.clip((wavelength_nm - wl_min) / (wl_max - wl_min), 0.0, 1.0)
    return int(round(frac * (n_bands - 1)))


def _select_bands(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select representative bands: green (~560nm), red (~670nm), NIR (~865nm), SWIR (~1600nm).
    Uses heuristic mapping across total band range.
    Returns: (green, red, nir, swir) as 2D arrays (H,W)
    """
    h, w, n_bands = cube.shape
    idx_green = _band_index_for_wavelength(n_bands, 560.0)
    idx_red   = _band_index_for_wavelength(n_bands, 670.0)
    idx_nir   = _band_index_for_wavelength(n_bands, 865.0)
    idx_swir  = _band_index_for_wavelength(n_bands, 1600.0)

    green = cube[:, :, idx_green]
    red   = cube[:, :, idx_red]
    nir   = cube[:, :, idx_nir]
    swir  = cube[:, :, idx_swir]

    # Ensure numeric stability
    green = np.nan_to_num(green, nan=0.0, posinf=0.0, neginf=0.0)
    red   = np.nan_to_num(red,   nan=0.0, posinf=0.0, neginf=0.0)
    nir   = np.nan_to_num(nir,   nan=0.0, posinf=0.0, neginf=0.0)
    swir  = np.nan_to_num(swir,  nan=0.0, posinf=0.0, neginf=0.0)

    return green, red, nir, swir


def compute_indices(cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NDVI, NDWI (McFeeters), and NDBI from the cube.
    Returns (ndvi, ndwi, ndbi) as float32 arrays in [-1,1].
    """
    green, red, nir, swir = _select_bands(cube)

    ndvi = (nir - red) / (nir + red + EPS)
    ndwi = (green - nir) / (green + nir + EPS)  # McFeeters 1996
    ndbi = (swir - nir) / (swir + nir + EPS)

    # Clip to sensible range
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)
    ndwi = np.clip(ndwi, -1.0, 1.0).astype(np.float32)
    ndbi = np.clip(ndbi, -1.0, 1.0).astype(np.float32)

    return ndvi, ndwi, ndbi


def _morph_clean(mask: np.ndarray, min_component_size: int = 20) -> np.ndarray:
    """Remove small components and smooth edges."""
    mask_u8 = mask.astype(np.uint8)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned[labels == lbl] = 1

    # Optional morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned.astype(bool)


def filter_manmade_anomalies(
    cube: np.ndarray,
    anomaly_mask: np.ndarray,
    ndvi_thresh: float = 0.2,
    ndwi_thresh: float = 0.0,
    ndbi_thresh: float = 0.0,
    min_component_size: int = 20,
) -> np.ndarray:
    """
    Constrain anomaly mask to man-made targets using heuristic spectral rules:
    - Non-vegetation: NDVI < ndvi_thresh
    - Non-water:      NDWI < ndwi_thresh
    - Built-up:       NDBI > ndbi_thresh

    Returns final boolean mask.
    """
    ndvi, ndwi, ndbi = compute_indices(cube)

    non_veg   = ndvi < ndvi_thresh
    non_water = ndwi < ndwi_thresh
    built_up  = ndbi > ndbi_thresh

    manmade_prior = non_veg & non_water & built_up

    final_mask = anomaly_mask & manmade_prior

    # Clean small speckles
    final_mask = _morph_clean(final_mask, min_component_size=min_component_size)

    # Debug summary
    total = anomaly_mask.size
    print("Man-made filter summary:")
    print(f"  NDVI< {ndvi_thresh:.2f}: {np.sum(non_veg)} ({np.sum(non_veg)/total*100:.2f}%)")
    print(f"  NDWI< {ndwi_thresh:.2f}: {np.sum(non_water)} ({np.sum(non_water)/total*100:.2f}%)")
    print(f"  NDBI> {ndbi_thresh:.2f}: {np.sum(built_up)} ({np.sum(built_up)/total*100:.2f}%)")
    print(f"  Raw anomalies: {np.sum(anomaly_mask)} ({np.sum(anomaly_mask)/total*100:.2f}%)")
    print(f"  Man-made constrained anomalies: {np.sum(final_mask)} ({np.sum(final_mask)/total*100:.2f}%)")

    return final_mask
