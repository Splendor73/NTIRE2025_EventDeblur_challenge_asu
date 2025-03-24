import os
import glob
import shutil

# Paths
blur_dir = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV/train/blur"
voxel_dir = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV/train/voxel_12bins"
backup_dir = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV/train/voxel_12bins_backup"

# Create backup directory
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# First, back up all voxel files
for voxel_file in glob.glob(os.path.join(voxel_dir, "*.npz")):
    shutil.copy(voxel_file, backup_dir)

# Get all blur files and their naming pattern
blur_files = sorted(glob.glob(os.path.join(blur_dir, "*.png")))
blur_prefixes = set()
for blur_file in blur_files:
    blur_name = os.path.basename(blur_file)
    prefix = '_'.join(blur_name.split('_')[:-1])  # Get everything except the last part (number)
    blur_prefixes.add(prefix)

print(f"Found {len(blur_prefixes)} different prefixes in blur files: {blur_prefixes}")

# For each blur prefix
for blur_prefix in blur_prefixes:
    # Get all blur files with this prefix
    blur_pattern_files = sorted(glob.glob(os.path.join(blur_dir, f"{blur_prefix}_*.png")))
    
    # For simplicity, assume we have a corresponding set of voxel files with a consistent prefix
    # Let's look for voxel files with the same numbering pattern
    for blur_file in blur_pattern_files:
        blur_name = os.path.basename(blur_file)
        number_part = blur_name.split('_')[-1].split('.')[0]  # Get just the number (00005, etc.)
        
        # Check if any voxel file exists with this number
        for voxel_prefix in os.listdir(backup_dir):
            voxel_prefix = os.path.splitext(voxel_prefix)[0].rsplit('_', 1)[0]  # Get prefix (TOITOI)
            if voxel_prefix:
                break
        
        old_voxel_path = os.path.join(voxel_dir, f"{voxel_prefix}_{number_part}.npz")
        new_voxel_path = os.path.join(voxel_dir, f"{blur_prefix}_{number_part}.npz")
        
        if os.path.exists(old_voxel_path):
            print(f"Renaming {old_voxel_path} to {new_voxel_path}")
            os.rename(old_voxel_path, new_voxel_path)
        else:
            print(f"Warning: No matching voxel file found for {blur_name}")