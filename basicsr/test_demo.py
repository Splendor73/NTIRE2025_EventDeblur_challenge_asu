#!/usr/bin/env python3

import os
import subprocess
import yaml
import shutil

def patch_test_model():
    """Temporarily patch the test model to handle missing keys"""
    model_file = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/basicsr/models/Test_image_event_restoration_model.py"
    backup_file = model_file + ".backup"
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_file):
        shutil.copy2(model_file, backup_file)
        print(f"Created backup of test model at {backup_file}")
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Replace the line with seq key access to handle missing key
    if "self.seq_name = data ['seq']" in content:
        content = content.replace(
            "self.seq_name = data ['seq']", 
            "self.seq_name = data.get('seq', 'unknown_seq')"
        )
        print("Patched Test_image_event_restoration_model.py to handle missing 'seq' key")
    
    # Replace the line with dataset_name key access
    if "if self.opt['dataset_name'] == 'REBlur':" in content:
        content = content.replace(
            "if self.opt['dataset_name'] == 'REBlur':", 
            "if self.opt.get('dataset_name', '') == 'REBlur':"
        )
        print("Patched Test_image_event_restoration_model.py to handle missing 'dataset_name' key")
    
    # Write the modified content back to the file
    with open(model_file, 'w') as f:
        f.write(content)

def restore_test_model():
    """Restore the original test model file"""
    model_file = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/basicsr/models/Test_image_event_restoration_model.py"
    backup_file = model_file + ".backup"
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, model_file)
        print(f"Restored original test model from {backup_file}")

def main():
    """
    Simple script to run the test.py with the correct configuration
    """
    # Define the path to the test.py script
    test_script = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/basicsr/test.py"
    
    # Define the path to the configuration file
    config_file = "/home/ypatel37/cvpr_pj_25/NTIRE2025_EventDeblur_challenge_asu/options/train/HighREV/KUnet_EFNet_HighREV_Deblur_voxel_test_3.yml"
    
    # Make sure the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        return
    
    try:
        # First patch the test model to handle missing keys
        patch_test_model()
        
        # Load and modify the YAML configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Fix the 'gt_size' parameter for the test dataset
        if 'datasets' in config and 'test' in config['datasets']:
            config['datasets']['test']['gt_size'] = None
            print("Added missing 'gt_size' parameter to test configuration")
        
        # Add the dataset_name parameter
        config['dataset_name'] = 'HighREV'
        print("Added 'dataset_name' parameter to configuration")
        
        # Change event_input_channels to 6 for the 6-bin voxel model
        if 'network_g' in config:
            config['network_g']['event_input_channels'] = 6
            print("Set event_input_channels to 6 for 6-bin voxel model")
        
        # Create a temporary config file with our changes
        temp_config_file = config_file + ".temp"
        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Build and execute the command with our temporary config
        cmd = ["python", test_script, "-opt", temp_config_file]
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        subprocess.run(cmd)
        
        # Clean up temporary file
        os.remove(temp_config_file)
        
    except Exception as e:
        print(f"Error in process: {e}")
    finally:
        # Restore the original test model file
        restore_test_model()

if __name__ == "__main__":
    main() 