import os
import shutil
import random

# -----------------------
# CONFIGURATION SETTINGS
# -----------------------
REMOVE_RANGE = (0, 50)   # Files to remove (start, end)
ADD_RANGE = (51, 100)    # Files to add (start, end)
REMOVE_BEFORE_ADD = True  # Set False if you only want to add files without removing

SOURCE_ROOT = "/scratch/ypatel37/database/HighREV"  # Source dataset
TARGET_ROOT = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV"
VOXEL_SOURCE_ROOT = "/scratch/ypatel37/database/HighREV_voxel"
VOXEL_TARGET_ROOT = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV_voxel"

DATASET_FOLDERS = ["train/blur", "train/sharp", "train/event", "val/blur", "val/sharp", "val/event"]
VOXEL_FOLDERS = ["train/voxel", "val/voxel"]

# -----------------------
# FUNCTION TO REMOVE FILES BASED ON RANGE
# -----------------------
def remove_files_by_range(directory, remove_range):
    """Removes files within the specified range in the directory."""
    if not os.path.exists(directory):
        print(f"❌ ERROR: Directory {directory} does not exist. Skipping...")
        return

    files = sorted(os.listdir(directory))
    start, end = remove_range

    if len(files) == 0:
        print(f"⚠️ Warning: No files to remove in {directory}")
        return

    files_to_remove = files[start:end]
    
    if len(files_to_remove) == 0:
        print(f"⚠️ Warning: No files found in range {start}-{end} for {directory}")
        return

    for file in files_to_remove:
        file_path = os.path.join(directory, file)
        os.remove(file_path)

    print(f"✅ Removed {len(files_to_remove)} files from {directory} (range: {start}-{end})")

# -----------------------
# FUNCTION TO COPY FILES BASED ON RANGE
# -----------------------
def copy_files_by_range(source_dir, target_dir, add_range):
    """Copies a range of files from the source to the target directory."""
    os.makedirs(target_dir, exist_ok=True)

    source_files = sorted(os.listdir(source_dir))
    start, end = add_range

    if start >= len(source_files):
        print(f"⚠️ Warning: No files available in range {start}-{end} for {source_dir}. Skipping...")
        return

    end = min(end, len(source_files))  # Ensure we don't go out of range
    files_to_copy = source_files[start:end]

    if not files_to_copy:
        print(f"⚠️ Warning: No files found in range {start}-{end} for {source_dir}")
        return

    for file in files_to_copy:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(target_dir, file)
        shutil.copy2(src_path, dest_path)

    print(f"✅ Copied {len(files_to_copy)} files from {source_dir} to {target_dir} (range: {start}-{end})")

# -----------------------
# MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    print("\n🚀 Starting dataset update...\n")

    if REMOVE_BEFORE_ADD:
        print("🗑 Removing old files first...")
        for folder in DATASET_FOLDERS:
            remove_files_by_range(os.path.join(TARGET_ROOT, folder), REMOVE_RANGE)

        for folder in VOXEL_FOLDERS:
            remove_files_by_range(os.path.join(VOXEL_TARGET_ROOT, folder), REMOVE_RANGE)

    print("\n📥 Adding new files...\n")
    for folder in DATASET_FOLDERS:
        copy_files_by_range(os.path.join(SOURCE_ROOT, folder), os.path.join(TARGET_ROOT, folder), ADD_RANGE)

    for folder in VOXEL_FOLDERS:
        copy_files_by_range(os.path.join(VOXEL_SOURCE_ROOT, folder), os.path.join(VOXEL_TARGET_ROOT, folder), ADD_RANGE)

    print("\n✅ Dataset update complete!\n")
