#!/usr/bin/env python3
"""
Quick test to see all geolocation metadata.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import extract_geolocation_metadata

def main():
    dataset_dir = r'E:\AIGR-S86462EUCLIDEAN TECHNOLOGIES_HYPERSPECTRAL\dataset'
    he5_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.he5')]
    
    if he5_files:
        he5_path = os.path.join(dataset_dir, he5_files[0])
        print(f"Analyzing: {he5_path}")
        
        geo_metadata = extract_geolocation_metadata(he5_path)
        print(f"\nFound {len(geo_metadata)} metadata items:")
        print("="*50)
        
        for key, val in geo_metadata.items():
            if isinstance(val, (list, tuple)):
                val_str = f"[{', '.join(map(str, val[:3]))}{'...' if len(val) > 3 else ''}]"
            elif hasattr(val, 'shape'):
                val_str = f"array shape {val.shape}"
            else:
                val_str = str(val)
            print(f"{key}: {val_str}")
    else:
        print("No HE5 files found")

if __name__ == '__main__':
    main()