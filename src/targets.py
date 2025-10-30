"""
Generate approximate spectral signatures for man-made materials over VNIR/SWIR bands.
If precise band centers are unavailable, assumes uniform spacing in wavelength.
"""

import numpy as np
from typing import Tuple


def _wavelengths(n_bands: int, data_type: str = 'BOTH') -> np.ndarray:
    if data_type.upper() == 'SWIR':
        wl_min, wl_max = 1000.0, 2500.0
    elif data_type.upper() == 'VNIR':
        wl_min, wl_max = 400.0, 1000.0
    else:  # BOTH
        wl_min, wl_max = 400.0, 2500.0
    return np.linspace(wl_min, wl_max, n_bands)


def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig -= sig.min()
    rng = sig.max() - sig.min() + 1e-12
    sig = sig / rng
    # Scale to realistic reflectance range [0.05, 0.6]
    return 0.05 + 0.55 * sig


def _asphalt(wl: np.ndarray) -> np.ndarray:
    # Dark, slightly increasing reflectance with wavelength
    base = 0.1 + 0.00015 * (wl - wl.min())
    return _normalize(base)


def _concrete(wl: np.ndarray) -> np.ndarray:
    # Brighter, gently varying spectrum
    base = 0.35 + 0.00005 * (wl - wl.min()) + 0.02 * np.sin((wl - wl.min()) / 200.0)
    return _normalize(base)


def _metal(wl: np.ndarray) -> np.ndarray:
    # Very low, flat reflectance
    base = 0.05 + 0.00001 * (wl - wl.min())
    return _normalize(base)


def _plastic(wl: np.ndarray) -> np.ndarray:
    # Introduce broad absorption dips around ~1210, ~1730, ~2310 nm when in range
    base = 0.25 + 0.00008 * (wl - wl.min())
    for center, width in [(1210, 80), (1730, 120), (2310, 120)]:
        dip = np.exp(-0.5 * ((wl - center) / width) ** 2)
        base -= 0.12 * dip
    return _normalize(base)


def get_manmade_signatures(cube: np.ndarray, data_type: str = 'BOTH') -> np.ndarray:
    """
    Build a small library of man-made spectral targets matching cube bands.
    Returns array of shape (T, bands).
    """
    n_bands = cube.shape[2]
    wl = _wavelengths(n_bands, data_type)

    targets = [
        _asphalt(wl),
        _concrete(wl),
        _metal(wl),
        _plastic(wl),
    ]

    S = np.stack(targets, axis=0).astype(np.float64)
    # Final normalization per target for ACE stability
    S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-12)
    return S
