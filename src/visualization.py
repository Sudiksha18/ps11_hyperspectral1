"""
Visualization utilities for anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2
import os
from datetime import datetime
import csv
import h5py
from typing import Dict

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: rasterio not available. GeoTIFF export will use alternative method.")
    RASTERIO_AVAILABLE = False

try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    print("Warning: GDAL not available. Using numpy-based TIFF export.")
    GDAL_AVAILABLE = False

def plot_anomaly_map(
    cube: np.ndarray,
    anomaly_mask: np.ndarray,
    k_masks: Optional[List[np.ndarray]] = None,
    rgb_preview: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    overlay_color: Tuple[float, float, float] = (1, 0, 0),  # Red
    overlay_alpha: float = 0.5,
    dpi: int = 300
) -> None:
    """
    Plot anomaly detection results with optional RGB preview and multi-k visualization.
    
    Args:
        cube: Original hyperspectral data
        anomaly_mask: Binary anomaly mask
        k_masks: Optional list of masks for different k values
        rgb_preview: Optional RGB preview image
        save_path: Path to save the plot
        overlay_color: RGB color for anomaly overlay
        overlay_alpha: Transparency of overlay
        dpi: DPI for saved figure
    """
    if rgb_preview is None:
        from .data_loader import create_rgb_preview
        rgb_preview = create_rgb_preview(cube, method='pca')
    
    # Create figure
    if k_masks is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        axes = [ax]
    
    # Plot RGB preview with overlay
    axes[0].imshow(rgb_preview)
    
    # Create overlay
    overlay = np.zeros_like(rgb_preview)
    overlay[anomaly_mask] = overlay_color
    
    axes[0].imshow(overlay, alpha=overlay_alpha * anomaly_mask.astype(float))
    axes[0].set_title('Anomaly Detection Results')
    axes[0].axis('off')
    
    if k_masks is not None:
        # Plot individual k results
        for i, (k_mask, ax) in enumerate(zip(k_masks, axes[1:])):
            ax.imshow(rgb_preview)
            
            overlay = np.zeros_like(rgb_preview)
            overlay[k_mask] = overlay_color
            
            ax.imshow(overlay, alpha=overlay_alpha * k_mask.astype(float))
            ax.set_title(f'k={i+1} Detection')
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_anomaly_overlay(
    rgb_preview: np.ndarray,
    anomaly_mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),  # BGR format for OpenCV
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay image with anomalies highlighted.
    
    Args:
        rgb_preview: RGB preview image
        anomaly_mask: Binary anomaly mask
        color: BGR color tuple for anomalies
        alpha: Transparency of overlay
        
    Returns:
        overlay: Image with anomalies highlighted
    """
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor((rgb_preview * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = bgr.copy()
    overlay[anomaly_mask] = color
    
    # Blend images
    output = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    
    return output


def export_geotiff_results(
    anomaly_mask: np.ndarray,
    output_dir: str = "results",
    filename_prefix: str = "anomaly_detection",
    submission_date: str = None
) -> str:
    """
    Export anomaly detection results as GeoTIFF for leaderboard submission.
    
    Args:
        anomaly_mask: Binary anomaly mask (True/False or 1/0)
        output_dir: Directory to save results
        filename_prefix: Prefix for output filename
        submission_date: Date for submission (e.g., "2025-10-06" or "2025-10-13")
        
    Returns:
        output_path: Path to saved GeoTIFF file
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        print("Installing rasterio for GeoTIFF export...")
        import subprocess
        subprocess.check_call(["pip", "install", "rasterio"])
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to boolean integers (0, 1)
    binary_mask = anomaly_mask.astype(np.uint8)
    
    # Set submission date
    if submission_date is None:
        current_date = datetime.now()
        # Use either Oct 6th or Oct 13th, 2025 based on current date
        if current_date.day <= 6:
            submission_date = "2025-10-06"
        else:
            submission_date = "2025-10-13"
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{filename_prefix}_{submission_date}_{timestamp}.tif"
    output_path = os.path.join(output_dir, filename)
    
    # Get image dimensions
    height, width = binary_mask.shape
    
    # Define coordinate system (using a generic geographic system)
    # In practice, you would use the actual coordinate system of your data
    crs = CRS.from_epsg(4326)  # WGS84
    
    # Create a basic transform (you should use actual geospatial coordinates)
    # This is a placeholder - replace with actual coordinates from your data
    transform = from_bounds(
        west=0.0, south=0.0, east=width, north=height, 
        width=width, height=height
    )
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress='lzw'  # Compression for smaller file size
    ) as dst:
        dst.write(binary_mask, 1)
        
        # Add metadata
        dst.update_tags(
            DESCRIPTION=f"Anomaly Detection Results - Boolean mask (0=normal, 1=anomaly)",
            SUBMISSION_DATE=submission_date,
            ALGORITHM="Adaptive_Mahalanobis_Distance",
            CREATED_BY="Enhanced_Hyperspectral_Anomaly_Detector",
            TIMESTAMP=datetime.now().isoformat()
        )
    
    print(f"✅ GeoTIFF exported successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {height} x {width} pixels")
    print(f"   Submission Date: {submission_date}")
    print(f"   Anomaly Count: {np.sum(binary_mask)} pixels ({np.sum(binary_mask)/binary_mask.size*100:.2f}%)")
    
    return output_path


def export_leaderboard_package(
    anomaly_mask: np.ndarray,
    cube_shape: Tuple[int, int, int],
    performance_stats: dict,
    output_dir: str = "leaderboard_submission",
    he5_path: str = None
) -> List[str]:
    """
    Export complete leaderboard submission package.
    
    Args:
        anomaly_mask: Binary anomaly mask
        cube_shape: Original hyperspectral cube shape
        performance_stats: Dictionary with performance statistics
        output_dir: Directory for submission files
        
    Returns:
        file_paths: List of generated file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    # Export for both submission dates
    dates = ["2025-10-06", "2025-10-13"]
    
    for date in dates:
        # Export GeoTIFF
        geotiff_path = export_geotiff_results(
            anomaly_mask, 
            output_dir=output_dir,
            filename_prefix="PRISMA_anomaly_detection",
            submission_date=date
        )
        file_paths.append(geotiff_path)
        
        # Export metadata file
        metadata_path = os.path.join(output_dir, f"metadata_{date}.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"PRISMA Hyperspectral Anomaly Detection Results\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Submission Date: {date}\n")
            f.write(f"Algorithm: Enhanced Adaptive Mahalanobis Distance\n")
            f.write(f"Original Data Shape: {cube_shape[0]} x {cube_shape[1]} x {cube_shape[2]}\n")
            f.write(f"Output Shape: {anomaly_mask.shape[0]} x {anomaly_mask.shape[1]}\n")
            f.write(f"Total Pixels: {anomaly_mask.size:,}\n")
            f.write(f"Anomaly Pixels: {np.sum(anomaly_mask):,}\n")
            f.write(f"Anomaly Percentage: {np.sum(anomaly_mask)/anomaly_mask.size*100:.2f}%\n\n")
            
            if performance_stats:
                f.write(f"Performance Statistics:\n")
                for key, value in performance_stats.items():
                    f.write(f"  {key}: {value}\n")
        
        file_paths.append(metadata_path)

        # Also export anomaly centroid coordinates (CSV) if geolocation available
        coords_path = os.path.join(output_dir, f"anomaly_centroids_{date}.csv")
        try:
            centroids = anomalies_to_geo_coords(anomaly_mask, he5_path=he5_path)
            # centroids is a list of dicts with keys: id, row, col, lon, lat
            with open(coords_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=['id', 'row', 'col', 'lon', 'lat'])
                writer.writeheader()
                for c in centroids:
                    writer.writerow(c)
            file_paths.append(coords_path)
        except Exception:
            # If geocoords are not available, skip
            pass

    print(f"\n🎯 Leaderboard submission package created!")
    print(f"   Directory: {output_dir}")
    print(f"   Files generated: {len(file_paths)}")

    return file_paths


def find_latlon_in_he5(he5_path: str) -> Dict[str, np.ndarray]:
        """
        Attempt to find latitude/longitude arrays inside an HE5 PRISMA file.
        Enhanced to also look for corner coordinates and geolocation metadata.

        Returns a dict {'lat': lat_array, 'lon': lon_array} if found, else empty dict.
        """
        res: Dict[str, np.ndarray] = {}
        if he5_path is None:
            return res
        try:
            with h5py.File(he5_path, 'r') as f:
                lat_ds = None
                lon_ds = None
                corner_coords = {}

                def recursive_search(group, path=""):
                    nonlocal lat_ds, lon_ds, corner_coords
                    for key, val in group.items():
                        current_path = f"{path}/{key}" if path else key
                        
                        if isinstance(val, h5py.Dataset):
                            lname = key.lower()
                            # Look for standard lat/lon arrays
                            if 'latitude' in lname or (('lat' in lname) and ('lon' not in lname and 'long' not in lname)):
                                if lat_ds is None:
                                    lat_ds = val[:]
                            if 'longitude' in lname or 'lon' in lname or 'long' in lname:
                                if lon_ds is None:
                                    lon_ds = val[:]
                            
                            # Look for corner coordinates or geolocation info
                            if any(corner_word in lname for corner_word in 
                                   ['corner', 'bound', 'extent', 'envelope', 'geolocation']):
                                try:
                                    corner_coords[current_path] = val[:]
                                except Exception:
                                    pass
                                    
                        elif isinstance(val, h5py.Group):
                            # Check group attributes for geolocation metadata
                            if val.attrs:
                                for attr_name, attr_val in val.attrs.items():
                                    attr_lower = attr_name.lower()
                                    if any(geo_word in attr_lower for geo_word in 
                                           ['corner', 'bound', 'lat', 'lon', 'north', 'south', 'east', 'west']):
                                        corner_coords[f"{current_path}@{attr_name}"] = attr_val
                            recursive_search(val, current_path)

                recursive_search(f)

                # If direct lat/lon arrays found, use them
                if lat_ds is not None and lon_ds is not None:
                    res['lat'] = lat_ds
                    res['lon'] = lon_ds
                else:
                    # Try to find corner coordinates and create approximate grid
                    res.update(_extract_corner_coords_he5(f, corner_coords))
                    
        except Exception:
            # If reading fails, return empty
            return {}

        return res


def _extract_corner_coords_he5(he5_file, corner_coords_dict):
    """
    Helper to extract corner coordinates from HE5 metadata and create lat/lon grids.
    """
    result = {}
    try:
        # Look for standard PRISMA corner coordinate patterns
        corners = {}
        
        # Search corner coordinate datasets and attributes
        for path, data in corner_coords_dict.items():
            path_lower = path.lower()
            if isinstance(data, np.ndarray) and data.size >= 4:
                # Try to interpret as corner coordinates
                flat_data = data.flatten()
                if len(flat_data) >= 4:
                    # Assume order: [north, south, east, west] or similar
                    if 'lat' in path_lower:
                        corners['lat_range'] = [float(np.min(flat_data)), float(np.max(flat_data))]
                    elif 'lon' in path_lower:
                        corners['lon_range'] = [float(np.min(flat_data)), float(np.max(flat_data))]
                    elif 'north' in path_lower:
                        corners['north'] = float(flat_data[0])
                    elif 'south' in path_lower:
                        corners['south'] = float(flat_data[0])
                    elif 'east' in path_lower:
                        corners['east'] = float(flat_data[0])
                    elif 'west' in path_lower:
                        corners['west'] = float(flat_data[0])
        
        # Also check root-level attributes for PRISMA corner coordinates
        try:
            for attr_name, attr_val in he5_file.attrs.items():
                attr_lower = attr_name.lower()
                if 'corner' in attr_lower or 'bound' in attr_lower:
                    if 'lat' in attr_lower and isinstance(attr_val, (float, int, np.number)):
                        if 'll' in attr_lower or 'lower' in attr_lower or 'south' in attr_lower:
                            corners['south'] = float(attr_val)
                        elif 'ur' in attr_lower or 'upper' in attr_lower or 'north' in attr_lower:
                            corners['north'] = float(attr_val)
                    elif ('lon' in attr_lower or 'long' in attr_lower) and isinstance(attr_val, (float, int, np.number)):
                        if 'll' in attr_lower or 'lower' in attr_lower or 'west' in attr_lower:
                            corners['west'] = float(attr_val)
                        elif 'ur' in attr_lower or 'upper' in attr_lower or 'east' in attr_lower:
                            corners['east'] = float(attr_val)
        except Exception:
            pass
        
        # Try to get image dimensions from SWIR data
        swir_shape = None
        try:
            swir_data = he5_file['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube']
            swir_shape = swir_data.shape[:2]  # (height, width)
        except Exception:
            try:
                # Alternative path structure
                for swir_path in ['/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube',
                                  '/SWIR_Cube', '/Data_Fields/SWIR_Cube']:
                    if swir_path in he5_file:
                        swir_shape = he5_file[swir_path].shape[:2]
                        break
            except Exception:
                pass
        
        # If we have corner info and image shape, create coordinate grids
        if swir_shape and (corners.get('lat_range') or 
                          (corners.get('north') and corners.get('south'))):
            height, width = swir_shape
            
            # Determine lat/lon bounds
            if corners.get('lat_range'):
                lat_min, lat_max = corners['lat_range']
            else:
                lat_min = corners.get('south', 0.0)
                lat_max = corners.get('north', 1.0)
                
            if corners.get('lon_range'):
                lon_min, lon_max = corners['lon_range']
            else:
                lon_min = corners.get('west', 0.0)
                lon_max = corners.get('east', 1.0)
            
            # Create coordinate grids
            lats = np.linspace(lat_max, lat_min, height)  # North to South
            lons = np.linspace(lon_min, lon_max, width)   # West to East
            
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            result['lat'] = lat_grid
            result['lon'] = lon_grid
            
            print(f"✅ Created coordinate grids from corner coordinates:")
            print(f"   Lat range: {lat_min:.4f} to {lat_max:.4f}")
            print(f"   Lon range: {lon_min:.4f} to {lon_max:.4f}")
            print(f"   Grid shape: {lat_grid.shape}")
            
    except Exception as e:
        print(f"Corner coordinate extraction failed: {e}")
        
    return result


def extract_prisma_corners_from_metadata(geo_metadata: dict) -> dict:
    """
    Extract PRISMA corner coordinates from the geolocation metadata.
    """
    corners = {}
    
    # Look for PRISMA-specific corner coordinate patterns
    for key, value in geo_metadata.items():
        key_lower = key.lower()
        
        if isinstance(value, (float, int, np.number)):
            # PRISMA corner coordinate patterns
            if 'llcorner_lat' in key_lower or ('lower' in key_lower and 'lat' in key_lower):
                corners['south'] = float(value)
            elif 'urcorner_lat' in key_lower or ('upper' in key_lower and 'lat' in key_lower):
                corners['north'] = float(value)
            elif 'llcorner_long' in key_lower or ('lower' in key_lower and ('long' in key_lower or 'lon' in key_lower)):
                corners['west'] = float(value)
            elif 'urcorner_long' in key_lower or ('upper' in key_lower and ('long' in key_lower or 'lon' in key_lower)):
                corners['east'] = float(value)
    
    # If we only have one corner (LL), try to estimate the opposite corner using image dimensions and typical pixel size
    if 'south' in corners and 'west' in corners and ('north' not in corners or 'east' not in corners):
        # PRISMA typical ground sample distance is about 30m
        # Image dimensions: 1202 x 173 pixels
        gsd_lat = 30.0 / 111320.0  # degrees per pixel latitude (approx)
        gsd_lon = 30.0 / (111320.0 * np.cos(np.radians(corners['south'])))  # degrees per pixel longitude
        
        height, width = 1202, 173  # known image dimensions
        
        if 'north' not in corners:
            corners['north'] = corners['south'] + (height * gsd_lat)
        if 'east' not in corners:
            corners['east'] = corners['west'] + (width * gsd_lon)
            
        print(f"Estimated missing corners using GSD:")
        print(f"  Estimated North: {corners.get('north'):.6f}")
        print(f"  Estimated East: {corners.get('east'):.6f}")
    
    return corners


def anomalies_to_geo_coords(anomaly_mask: np.ndarray, he5_path: str = None, transform=None, crs: str = 'EPSG:4326', geo_metadata: dict = None) -> List[Dict[str, float]]:
        """
        Compute centroids of connected anomaly components and map to geographic coordinates.

        Args:
            anomaly_mask: 2D binary mask
            he5_path: Optional path to HE5 file to extract lat/lon arrays (preferred)
            transform: Optional affine transform (rasterio) to map pixel->world
            crs: CRS string
            geo_metadata: Optional geolocation metadata dict from data_loader

        Returns:
            List of dicts with keys: id, row, col, lon, lat
        """
        # Ensure binary
        mask = (anomaly_mask > 0).astype(np.uint8)
        h, w = mask.shape

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        coords = []

        # Try to fetch lat/lon arrays if provided
        latlon = {}
        if he5_path is not None:
            try:
                latlon = find_latlon_in_he5(he5_path)
                print(f"Direct HE5 search found: {list(latlon.keys())}")
            except Exception as e:
                print(f"Direct HE5 search failed: {e}")
                latlon = {}
        
        # If no lat/lon arrays found but we have geo_metadata, check for geolocation datasets
        if not latlon and geo_metadata:
            try:
                # Look for geolocation lat/lon arrays in metadata
                for key, value in geo_metadata.items():
                    if isinstance(value, np.ndarray):
                        if 'latitude' in key.lower() and value.shape == (h, w):
                            latlon['lat'] = value
                            print(f"Found latitude array in metadata: {key}, shape: {value.shape}")
                        elif ('longitude' in key.lower() or 'lon' in key.lower()) and value.shape == (h, w):
                            latlon['lon'] = value  
                            print(f"Found longitude array in metadata: {key}, shape: {value.shape}")
                
                # If still no arrays but we have corners, create coordinate grid
                if not latlon:
                    corners = extract_prisma_corners_from_metadata(geo_metadata)
                    print(f"Extracted corners from metadata: {corners}")
                    
                    # Force creation of coordinate grid even with only 2 corners by estimating the others
                    if 'south' in corners and 'west' in corners:
                        print(f"Creating coordinate grid from PRISMA metadata corners:")
                        print(f"  South: {corners.get('south')}, West: {corners.get('west')}")
                        
                        # Create coordinate grids using the mask shape
                        height, width = h, w
                        lat_min = corners.get('south')
                        lat_max = corners.get('north') 
                        lon_min = corners.get('west')
                        lon_max = corners.get('east')
                        
                        if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
                            # Create coordinate grids
                            lats = np.linspace(lat_max, lat_min, height)  # North to South
                            lons = np.linspace(lon_min, lon_max, width)   # West to East
                            
                            lon_grid, lat_grid = np.meshgrid(lons, lats)
                            
                            latlon['lat'] = lat_grid
                            latlon['lon'] = lon_grid
                            
                            print(f"✅ Created coordinate grids from PRISMA corners:")
                            print(f"   Lat range: {lat_min:.6f} to {lat_max:.6f}")
                            print(f"   Lon range: {lon_min:.6f} to {lon_max:.6f}")
                            print(f"   Grid shape: {lat_grid.shape}")
                            print(f"   Sample coordinates: lat[0,0]={lat_grid[0,0]:.6f}, lon[0,0]={lon_grid[0,0]:.6f}")
            except Exception as e:
                print(f"Failed to create coordinate grid from metadata: {e}")
                import traceback
                traceback.print_exc()

        for lbl in range(1, num_labels):
            c = centroids[lbl]  # (cx, cy)
            cx, cy = float(c[0]), float(c[1])
            col = int(round(cx))
            row = int(round(cy))
            lon = None
            lat = None

            # If lat/lon arrays available and match shape, grab value
            if 'lat' in latlon and 'lon' in latlon:
                lat_arr = latlon['lat']
                lon_arr = latlon['lon']
                try:
                    if lat_arr.shape == mask.shape:
                        lat = float(lat_arr[row, col])
                        lon = float(lon_arr[row, col])
                    else:
                        # try transposed
                        if lat_arr.T.shape == mask.shape:
                            lat = float(lat_arr.T[row, col])
                            lon = float(lon_arr.T[row, col])
                except Exception:
                    lat = None
                    lon = None

            # If transform provided (rasterio), map pixel to coords
            if (lat is None or lon is None) and transform is not None:
                try:
                    from rasterio.transform import xy
                    x, y = xy(transform, row, col, offset='center')
                    lon, lat = float(x), float(y)
                except Exception:
                    pass

            coords.append({'id': int(lbl), 'row': int(row), 'col': int(col), 'lon': lon, 'lat': lat})

        return coords

def export_geotiff(
    anomaly_mask: np.ndarray,
    output_path: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    crs: str = 'EPSG:4326'
) -> str:
    """
    Export anomaly detection results as GeoTIFF for leaderboard submission.
    
    Args:
        anomaly_mask: Boolean anomaly mask (0=normal, 1=anomaly)
        output_path: Path to save GeoTIFF file
        bounds: Spatial bounds (minx, miny, maxx, maxy)
        crs: Coordinate reference system
        
    Returns:
        Path to created GeoTIFF file
    """
    # Convert boolean mask to uint8 (0,1)
    geotiff_data = anomaly_mask.astype(np.uint8)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if RASTERIO_AVAILABLE:
        # Use rasterio for proper GeoTIFF export
        height, width = geotiff_data.shape
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (0, 0, width, height)
        
        # Create transform
        transform = from_bounds(*bounds, width, height)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=geotiff_data.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(geotiff_data, 1)
            
    elif GDAL_AVAILABLE:
        # Use GDAL for GeoTIFF export
        height, width = geotiff_data.shape
        
        # Create dataset
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
        
        # Set geotransform
        if bounds is None:
            bounds = (0, 0, width, height)
        geotransform = [bounds[0], (bounds[2]-bounds[0])/width, 0, 
                       bounds[3], 0, -(bounds[3]-bounds[1])/height]
        dataset.SetGeoTransform(geotransform)
        
        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(crs.split(':')[1]) if ':' in crs else 4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.WriteArray(geotiff_data)
        band.SetNoDataValue(255)
        
        # Close dataset
        dataset = None
        
    else:
        # Fallback: Save as TIFF using PIL/OpenCV
        try:
            from PIL import Image
            # Convert to PIL Image and save
            img = Image.fromarray(geotiff_data * 255, mode='L')
            img.save(output_path.replace('.tif', '_fallback.tif'))
            print(f"Warning: Saved as regular TIFF (no geo-referencing): {output_path}")
        except ImportError:
            # Final fallback: use OpenCV
            cv2.imwrite(output_path.replace('.tif', '_fallback.png'), geotiff_data * 255)
            print(f"Warning: Saved as PNG (no geo-referencing): {output_path}")
    
    print(f"✅ GeoTIFF exported: {output_path}")
    return output_path

def plot_multi_k_bw_detection(
    k_masks: List[np.ndarray],
    k_values: List[float],
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES Multi-K Anomaly Detection",
    dpi: int = 300
) -> str:
    """
    Create a comprehensive black and white visualization showing all k-value detections.
    
    Args:
        k_masks: List of binary masks for each k value
        k_values: List of k values corresponding to masks
        save_path: Path to save the visualization
        title: Title for the plot
        dpi: DPI for saved figure
        
    Returns:
        Path to saved visualization
    """
    n_k = len(k_masks)
    height, width = k_masks[0].shape
    
    # Create a comprehensive layout
    if n_k <= 4:
        rows, cols = 2, 2
    elif n_k <= 6:
        rows, cols = 2, 3
    elif n_k <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4
    
    # Create figure with black background
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold')
    
    # Individual k-value detections
    for i, (mask, k) in enumerate(zip(k_masks, k_values)):
        if i >= rows * cols:
            break
            
        ax = plt.subplot(rows, cols, i + 1)
        
        # Create black and white image (white = anomaly, black = normal)
        bw_image = mask.astype(np.uint8) * 255
        
        # Display
        ax.imshow(bw_image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'k = {k} ({np.sum(mask)} anomalies)', 
                    color='white', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
    
    # Remove unused subplots
    for i in range(n_k, rows * cols):
        if i < rows * cols:
            ax = plt.subplot(rows, cols, i + 1)
            ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save with black background
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='black', edgecolor='white')
    plt.close()
    
    print(f"✅ Multi-k black & white detection saved: {save_path}")
    return save_path

def plot_combined_multi_k_bw(
    k_masks: List[np.ndarray],
    k_values: List[float],
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES Combined Multi-K Detection Map",
    dpi: int = 300
) -> str:
    """
    Create a single combined black and white image showing intensity levels for different k detections.
    
    Args:
        k_masks: List of binary masks for each k value
        k_values: List of k values corresponding to masks
        save_path: Path to save the visualization
        title: Title for the plot
        dpi: DPI for saved figure
        
    Returns:
        Path to saved visualization
    """
    height, width = k_masks[0].shape
    
    # Create intensity map based on detection level
    # Higher k values get higher intensity (brighter white)
    combined_map = np.zeros((height, width), dtype=np.float32)
    
    # Weight by inverse k value (smaller k = more significant = brighter)
    max_k = max(k_values)
    for mask, k in zip(k_masks, k_values):
        # Inverse weighting: smaller k values get higher weight
        weight = (max_k - k + 0.1) / max_k
        combined_map += mask.astype(np.float32) * weight
    
    # Normalize to 0-255 range
    if np.max(combined_map) > 0:
        combined_map = (combined_map / np.max(combined_map)) * 255
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
    
    # Combined intensity map
    im1 = ax1.imshow(combined_map, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Combined Detection Intensity Map\n(Brighter = More Significant)', 
                 color='white', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.ax.yaxis.label.set_color('white')
    cbar1.set_label('Detection Intensity', color='white')
    
    # Binary union of all detections
    union_mask = np.zeros_like(k_masks[0], dtype=bool)
    for mask in k_masks:
        union_mask |= mask
    
    binary_image = union_mask.astype(np.uint8) * 255
    ax2.imshow(binary_image, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f'All Detections Combined\n({np.sum(union_mask)} total anomalies)', 
                 color='white', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Style both axes
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
    
    # Overall title
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save with black background
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='black', edgecolor='white')
    plt.close()
    
    print(f"✅ Combined multi-k black & white detection saved: {save_path}")
    return save_path


def save_union_bw_image(
    anomaly_mask: np.ndarray,
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES All Anomalies (Union)",
    dpi: int = 300
) -> str:
    """
    Save a single black & white image showing ALL anomalies (union mask only).
    White = anomaly, Black = normal.
    """
    bw_image = anomaly_mask.astype(np.uint8) * 255
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.imshow(bw_image, cmap='gray', vmin=0, vmax=255)
    plt.title(title, color='white', fontsize=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='black', edgecolor='white')
    plt.close()
    print(f"✅ Union (all anomalies) black & white image saved: {save_path}")
    return save_path


def save_score_heatmap(
    scores: np.ndarray,
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES ACE Score Heatmap",
    cmap: str = 'inferno',
    dpi: int = 300,
    clip_percentiles: tuple = (1, 99)
) -> str:
    """
    Save a color heatmap of per-pixel scores (e.g., ACE or distances).
    Automatically clips extreme values using percentiles for contrast.
    """
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        raise ValueError("No valid scores to visualize")
    lo, hi = np.percentile(valid, clip_percentiles)
    lo = float(lo)
    hi = float(hi) if hi > lo else lo + 1e-6

    plt.figure(figsize=(10, 6), facecolor='black')
    im = plt.imshow(scores, cmap=cmap, vmin=lo, vmax=hi)
    plt.title(title, color='white', fontsize=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    cbar.set_label('Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='black', edgecolor='white')
    plt.close()
    print(f"✅ Score heatmap saved: {save_path}")
    return save_path


def save_multi_k_single_image(
    k_masks: List[np.ndarray],
    k_values: List[float],
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES Multi-K Detection (Counts)",
    dpi: int = 300,
    cmap: str = 'viridis'
) -> str:
    """
    Save a single image encoding how many k-values flagged each pixel (0..N).
    This shows all k results in one color image instead of a union or grid.
    """
    # Compute count map
    count_map = np.zeros_like(k_masks[0], dtype=np.int16)
    for m in k_masks:
        count_map += m.astype(np.int16)

    # Display
    plt.figure(figsize=(10, 6), facecolor='black')
    im = plt.imshow(count_map, cmap=cmap, vmin=0, vmax=len(k_masks))
    plt.title(title + f"\nCounts per pixel (0..{len(k_masks)})", color='white', fontsize=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    cbar.set_label('Number of k-values that flagged pixel')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='black', edgecolor='white')
    plt.close()
    print(f"✅ Single multi-k image (counts) saved: {save_path}")
    return save_path


def save_multi_k_rgb_overlay(
    rgb_preview: np.ndarray,
    k_masks: List[np.ndarray],
    save_path: str,
    title: str = "EUCLIDEAN_TECHNOLOGIES Multi-K Detection (RGB Overlay)",
    alpha: float = 0.6,
    cmap: str = 'plasma',
    dpi: int = 300
) -> str:
    """
    Save a single RGB image overlaying the multi-k count map on the RGB preview.
    """
    # Ensure RGB in [0,1]
    rgb = np.clip(np.nan_to_num(rgb_preview, nan=0.0), 0.0, 1.0)

    # Count map 0..N
    count_map = np.zeros(k_masks[0].shape, dtype=np.int16)
    for m in k_masks:
        count_map += m.astype(np.int16)

    # Normalize to 0..1 for colormap
    if len(k_masks) > 0:
        norm = count_map.astype(np.float32) / float(len(k_masks))
    else:
        norm = count_map.astype(np.float32)

    # Colormap to RGB
    cm = plt.get_cmap(cmap)
    overlay_rgb = cm(norm)[..., :3]  # drop alpha

    # Alpha blend
    blended = (1 - alpha) * rgb + alpha * overlay_rgb
    blended = np.clip(blended, 0.0, 1.0)

    # Plot and save
    plt.figure(figsize=(10, 6))
    plt.imshow(blended)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Multi-k RGB overlay saved: {save_path}")
    return save_path
