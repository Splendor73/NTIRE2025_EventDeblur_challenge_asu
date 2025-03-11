import os
import shutil
import argparse

def copy_files(source_dir, target_dir, num_files):
    """ Copies a subset of files from source to target directory. """
    os.makedirs(target_dir, exist_ok=True)
    
    # List all files
    files = sorted(os.listdir(source_dir))[:num_files]
    
    if not files:
        print(f"❌ ERROR: No files found in {source_dir}")
        return

    print(f"✅ Copying {len(files)} files from {source_dir} to {target_dir}")
    
    for file_name in files:
        src_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(target_dir, file_name)
        shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a subset of dataset files.")
    parser.add_argument("--source", type=str, required=True, help="Source directory")
    parser.add_argument("--target", type=str, required=True, help="Target directory")
    parser.add_argument("--num_files", type=int, required=True, help="Number of files to copy")

    args = parser.parse_args()
    
    copy_files(args.source, args.target, args.num_files)
