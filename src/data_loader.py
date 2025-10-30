"""
Utility functions for loading and preprocessing hyperspectral data from HE5 files.
Updated for PRISMA dataset compatibility.
"""

import h5py
import numpy as np
from typing import Tuple, Optional, Dict

def extract_geolocation_metadata(filepath: str) -> Dict:
    """
    Extract geolocation metadata from PRISMA HE5 file.
    
    Args:
        filepath: Path to the HE5 file
        
    Returns:
        dict: Geolocation metadata including corner coordinates if available
    """
    metadata = {}
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Look for geolocation metadata in attributes
            def search_attributes(group, path=""):
                for attr_name, attr_val in group.attrs.items():
                    attr_lower = attr_name.lower()
                    if any(geo_word in attr_lower for geo_word in 
                           ['corner', 'bound', 'lat', 'lon', 'north', 'south', 'east', 'west',
                            'coordinate', 'geographic', 'projection', 'crs']):
                        metadata[f"{path}/{attr_name}" if path else attr_name] = attr_val
                
                for key, val in group.items():
                    if isinstance(val, h5py.Group):
                        current_path = f"{path}/{key}" if path else key
                        search_attributes(val, current_path)
            
            search_attributes(f)
            
            # Look for specific PRISMA geolocation datasets
            geoloc_paths = [
                '/HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields',
                '/Geolocation_Fields',
                '/Geolocation'
            ]
            
            for geoloc_path in geoloc_paths:
                if geoloc_path in f:
                    geoloc_group = f[geoloc_path]
                    for key, val in geoloc_group.items():
                        if isinstance(val, h5py.Dataset):
                            try:
                                # For lat/lon arrays, load the actual data
                                if key.lower() in ['latitude', 'longitude']:
                                    data_array = val[:]
                                    metadata[f"geoloc/{key}"] = data_array
                                    print(f"Loaded geolocation array: {key}, shape: {data_array.shape}")
                                else:
                                    metadata[f"geoloc/{key}"] = val[:]
                            except Exception:
                                metadata[f"geoloc/{key}"] = f"Dataset shape: {val.shape}"
            
            # Extract corner coordinates from metadata if available
            corner_coords = extract_corner_coordinates(metadata)
            if corner_coords:
                metadata.update(corner_coords)
                
    except Exception as e:
        print(f"Warning: Could not extract geolocation metadata: {e}")
    
    return metadata


def extract_corner_coordinates(metadata: Dict) -> Dict:
    """
    Extract corner coordinates from metadata dict.
    
    Args:
        metadata: Dictionary of geolocation metadata
        
    Returns:
        dict: Corner coordinates if found
    """
    corners = {}
    
    # Look for corner coordinate patterns in metadata
    for key, value in metadata.items():
        key_lower = key.lower()
        
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                if hasattr(value, 'flatten'):
                    flat_vals = value.flatten()
                else:
                    flat_vals = np.array(value).flatten()
                
                # Look for latitude bounds
                if any(lat_word in key_lower for lat_word in ['lat', 'latitude']):
                    if len(flat_vals) >= 2:
                        corners['lat_min'] = float(np.min(flat_vals))
                        corners['lat_max'] = float(np.max(flat_vals))
                
                # Look for longitude bounds  
                elif any(lon_word in key_lower for lon_word in ['lon', 'longitude']):
                    if len(flat_vals) >= 2:
                        corners['lon_min'] = float(np.min(flat_vals))
                        corners['lon_max'] = float(np.max(flat_vals))
                
                # Look for specific corner names
                elif 'north' in key_lower:
                    corners['north'] = float(flat_vals[0])
                elif 'south' in key_lower:
                    corners['south'] = float(flat_vals[0])
                elif 'east' in key_lower:
                    corners['east'] = float(flat_vals[0])
                elif 'west' in key_lower:
                    corners['west'] = float(flat_vals[0])
                    
            except Exception:
                continue
    
    return corners

def load_hyperspectral_data(
    filepath: str,
    normalize: bool = True,
    data_type: str = 'BOTH',  # 'SWIR', 'VNIR', or 'BOTH'
    subset_bands: Optional[slice] = None
) -> np.ndarray:
    """
    Load hyperspectral data from PRISMA HE5 file.
    
    Args:
        filepath: Path to the HE5 file
        normalize: Whether to normalize the data
        data_type: Which spectral data to load ('SWIR', 'VNIR', or 'BOTH')
        subset_bands: Optional slice to select specific bands
        
    Returns:
        cube: Hyperspectral data array (height, width, bands)
    """
    print(f"Loading PRISMA data from: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Define paths for different cubes
        swir_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'
        vnir_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'
        
        # Load data based on requested type
        if data_type == 'SWIR':
            if swir_path in f:
                data = f[swir_path][:]
            else:
                raise ValueError("SWIR data not found in file")
        elif data_type == 'VNIR':
            if vnir_path in f:
                data = f[vnir_path][:]
            else:
                raise ValueError("VNIR data not found in file")
        elif data_type == 'BOTH':
            # Load both and concatenate along spectral dimension
            swir_data = None
            vnir_data = None
            
            if swir_path in f:
                swir_data = f[swir_path][:]
                print(f"SWIR data shape: {swir_data.shape}")
            
            if vnir_path in f:
                vnir_data = f[vnir_path][:]
                print(f"VNIR data shape: {vnir_data.shape}")
            
            if swir_data is not None and vnir_data is not None:
                # Concatenate along the spectral dimension (axis=2)
                data = np.concatenate([vnir_data, swir_data], axis=2)
                print(f"Combined VNIR+SWIR shape: {data.shape}")
            elif swir_data is not None:
                data = swir_data
                print("Using SWIR data only")
            elif vnir_data is not None:
                data = vnir_data
                print("Using VNIR data only")
            else:
                raise ValueError("No hyperspectral data found in file")
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    # Apply band subset if specified
    if subset_bands is not None:
        data = data[:, :, subset_bands]
        print(f"Subset applied, new shape: {data.shape}")
    
    # Normalize if requested
    if normalize:
        print("Normalizing data...")
        # Handle invalid values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to [0, 1] range
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        
        if data_range > 0:
            data = (data - data_min) / data_range
        
        print(f"Normalized data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    print(f"Final loaded cube shape: {data.shape}")
    return data

def create_rgb_preview(
    cube: np.ndarray,
    red_band: int = -1,
    green_band: int = -1,
    blue_band: int = -1,
    method: str = 'default'
) -> np.ndarray:
    """
    Create RGB preview image from hyperspectral data.
    
    Args:
        cube: Hyperspectral data array (height, width, bands)
        red_band: Band index for red channel (-1 for auto)
        green_band: Band index for green channel (-1 for auto)  
        blue_band: Band index for blue channel (-1 for auto)
        method: Method for creating RGB ('default', 'pca', or 'max_variance')
        
    Returns:
        rgb: RGB image array (height, width, 3)
    """
    height, width, n_bands = cube.shape
    
    if method == 'pca':
        # Use PCA to create RGB preview
        from sklearn.decomposition import PCA
        
        # Reshape for PCA
        cube_flat = cube.reshape(-1, n_bands)
        
        # Handle NaN values
        valid_mask = ~np.isnan(cube_flat).any(axis=1)
        cube_valid = cube_flat[valid_mask]
        
        if len(cube_valid) == 0:
            # Fallback to zeros if no valid data
            return np.zeros((height, width, 3))
        
        # Apply PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(cube_valid)
        
        # Create full result array
        pca_full = np.zeros((len(cube_flat), 3))
        pca_full[valid_mask] = pca_result
        
        # Reshape back to image
        rgb = pca_full.reshape(height, width, 3)
        
        # Normalize each channel
        for i in range(3):
            channel = rgb[:, :, i]
            channel_min, channel_max = np.nanmin(channel), np.nanmax(channel)
            if channel_max > channel_min:
                rgb[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
    
    else:
        # Default method: select bands based on spectral characteristics
        if red_band == -1:
            # Select band from the red portion of spectrum (assuming higher band numbers = longer wavelengths)
            red_band = min(n_bands - 1, int(n_bands * 0.85))  # Near end for red
        
        if green_band == -1:
            # Select band from green portion
            green_band = int(n_bands * 0.5)  # Middle for green
        
        if blue_band == -1:
            # Select band from blue portion
            blue_band = int(n_bands * 0.15)  # Near beginning for blue
        
        # Ensure band indices are valid
        red_band = np.clip(red_band, 0, n_bands - 1)
        green_band = np.clip(green_band, 0, n_bands - 1)
        blue_band = np.clip(blue_band, 0, n_bands - 1)
        
        print(f"RGB bands selected: R={red_band}, G={green_band}, B={blue_band}")
        
        # Extract selected bands
        rgb = np.stack([
            cube[:, :, red_band],
            cube[:, :, green_band], 
            cube[:, :, blue_band]
        ], axis=-1)
    
    # Handle NaNs and ensure valid range
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def get_dataset_info(filepath: str) -> dict:
    """
    Get information about the PRISMA dataset.
    
    Args:
        filepath: Path to the HE5 file
        
    Returns:
        info: Dictionary containing dataset information
    """
    info = {}
    
    with h5py.File(filepath, 'r') as f:
        # List all available datasets
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'size_mb': obj.size * obj.dtype.itemsize / (1024**2)
                }
        
        f.visititems(visit_func)
    
    return info