#!/bin/bash

# Directory containing the voxel files
VOXEL_DIR="/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/datasets/HighREV_test/voxel"

# Change to the voxel directory
cd "$VOXEL_DIR" || { echo "Error: Directory not found"; exit 1; }

# Find all files with ._ prefix and rename them
echo "Renaming files with ._ prefix..."
count=0

# For npz files
for file in ._*.npz; do
    # Check if the file exists (to handle case where no matches are found)
    if [ -f "$file" ]; then
        # Extract the filename without the ._ prefix
        newname="${file#._}"
        
        # Handle case where the file without prefix already exists
        if [ -f "$newname" ]; then
            echo "Warning: $newname already exists, skipping $file"
        else
            mv "$file" "$newname"
            echo "Renamed: $file -> $newname"
            count=$((count+1))
        fi
    fi
done

echo "Renamed $count files."