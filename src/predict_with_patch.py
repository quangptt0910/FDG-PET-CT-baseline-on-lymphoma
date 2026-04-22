import os
import sys
import torch
import numpy as np

# Monkeypatch torch.load BEFORE importing nnunetv2
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Ensure numpy multiarray scalar is allowed if weights_only is True (belt and suspenders)
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except:
    pass

from nnunetv2.inference.predict_from_raw_data import predict_entry_point

def save_biomarker_json(output_dir):
    import nibabel as nib
    import numpy as np
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    mask_files = list(output_path.glob("*.nii.gz"))
    
    if not mask_files:
        return

    for mask_path in mask_files:
        try:
            case_id = mask_path.name.replace(".nii.gz", "")
            img = nib.load(mask_path)
            data = img.get_fdata()
            
            # 1. Voxel Count
            voxel_count = int(np.sum(data == 1))
            
            # 2. Volume in ml
            spacing_vector = img.header.get_zooms()
            spacing_prod = float(np.prod(spacing_vector))
            volume_ml = float((voxel_count * spacing_prod) / 1000.0)
            
            # 3. Construct JSON structure
            report = {
                "case_id": case_id,
                "biomarkers": {
                    "total_lesion_voxels": int(voxel_count),
                    "tmtv_ml": round(volume_ml, 4),
                    "volume_unit": "ml"
                },
                "spatial_metadata": {
                    "voxel_spacing_mm": [round(float(s), 4) for s in spacing_vector],
                    "image_dimension": [int(d) for d in img.shape]
                }
            }
            
            # Save to JSON
            json_path = output_path / f"{case_id}_biomarkers.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            print(f"Processed {case_id}: {volume_ml:.3f} ml -> {json_path.name}")
            
        except Exception as e:
            print(f"Failed to generate JSON for {mask_path.name}: {e}")

if __name__ == '__main__':
    # Parse args manually to get the output directory for the report
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-o', '--o', '-output', '--output', type=str, required=False)
    args, _ = parser.parse_known_args()
    
    # Run original nnU-Net prediction
    predict_entry_point()
    
    # Run our biomarker JSON generation if output dir is known
    if args.o:
        save_biomarker_json(args.o)
