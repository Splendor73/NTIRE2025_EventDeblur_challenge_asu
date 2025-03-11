#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import glob

def copy_dataset_subset(source_dir, target_dir, start_idx, end_idx):
    """
    Copy a subset of files from source_dir to target_dir in batches.
    """
    os.makedirs(target_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(source_dir, "*.npz")))

    if not files:
        print(f"❌ ERROR: No .npz files found in {source_dir}")
        return

    # Limit indices
    start_idx = max(0, start_idx)
    end_idx = min(len(files) - 1, end_idx)

    subset_files = files[start_idx:end_idx + 1]
    
    print(f"📦 Copying {len(subset_files)} files from {source_dir} to {target_dir}")

    for i, file_path in enumerate(subset_files):
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_dir, file_name)

        # Only copy if file does not exist
        if not os.path.exists(target_path):
            shutil.copy2(file_path, target_path)

        if (i + 1) % 10 == 0 or i == len(subset_files) - 1:
            print(f"✅ Copied {i+1}/{len(subset_files)} files")

def main():
    parser = argparse.ArgumentParser(description="Copy a subset of dataset files")
    parser.add_argument("--source", type=str, required=True, help="Source directory with voxel files")
    parser.add_argument("--target", type=str, required=True, help="Target directory")
    parser.add_argument("--start", type=int, default=0, help="Starting index (inclusive)")
    parser.add_argument("--end", type=int, default=49, help="Ending index (inclusive)")
    
    args = parser.parse_args()
    
    copy_dataset_subset(args.source, args.target, args.start, args.end)

if __name__ == "__main__":
    main()
