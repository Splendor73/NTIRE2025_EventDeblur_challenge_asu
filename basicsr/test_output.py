import logging
import torch
import os
import numpy as np
from os import path as osp
from tqdm import tqdm
import cv2
import glob

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


class TestDataset:
    """Custom dataset for testing without GT images."""
    def __init__(self, dataroot, dataroot_voxel, io_backend=None):
        self.dataroot = dataroot
        self.dataroot_voxel = dataroot_voxel
        
        # Get all blur image files
        self.blur_paths = sorted([
            os.path.join(dataroot, 'blur', f) 
            for f in os.listdir(os.path.join(dataroot, 'blur'))
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Extract sequence names from blur paths
        self.seq_names = [os.path.splitext(os.path.basename(p))[0] for p in self.blur_paths]
        
        # Scan voxel directory
        self.voxel_files = {}
        self._scan_voxel_directory()
        
        print(f"Found {len(self.blur_paths)} test images.")
        print(f"Found {len(self.voxel_files)} voxel files.")

    def _scan_voxel_directory(self):
        """Scan voxel directory and map blur images to voxel files."""
        voxel_dir = os.path.join(self.dataroot_voxel, 'voxel')
        print(f"Scanning voxel directory: {voxel_dir}")
        
        if not os.path.exists(voxel_dir):
            print(f"WARNING: Voxel directory not found: {voxel_dir}")
            return
        
        # Find all npz files in the voxel directory
        voxel_files = glob.glob(os.path.join(voxel_dir, "**/*.npz"), recursive=True)
        
        if len(voxel_files) == 0:
            print(f"WARNING: No .npz files found in {voxel_dir}")
            return
        
        print(f"Found {len(voxel_files)} .npz files in voxel directory")
        
        # List a few voxel files to understand naming pattern
        if len(voxel_files) > 0:
            print("Sample voxel files:")
            for i in range(min(5, len(voxel_files))):
                print(f"  {os.path.basename(voxel_files[i])}")
        
        # Create mapping between sequence names and voxel files
        for seq_name in self.seq_names:
            # Try to find matching voxel file
            base_name = seq_name.split('_')[0]  # Get the base name (e.g., "logo" from "logo_ethzurich_00017")
            
            # Try different patterns to match voxel files
            candidates = []
            
            # Pattern 1: Exact match
            candidates.extend(glob.glob(os.path.join(voxel_dir, f"{seq_name}*.npz")))
            
            # Pattern 2: Base name match
            candidates.extend(glob.glob(os.path.join(voxel_dir, f"{base_name}*.npz")))
            
            # If multiple matches, prefer the ones with sequence number closest to the blur image
            if candidates:
                self.voxel_files[seq_name] = candidates[0]
                if len(candidates) > 1:
                    print(f"Multiple voxel matches for {seq_name}, using: {os.path.basename(candidates[0])}")
            else:
                print(f"WARNING: No matching voxel file found for {seq_name}")

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        # Load blur image
        blur_path = self.blur_paths[idx]
        seq_name = self.seq_names[idx]
        
        # Load blur image
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB) / 255.0
        blur_img = torch.from_numpy(blur_img.transpose(2, 0, 1)).float()
        
        # Get voxel path if available
        voxel_tensor = None
        if seq_name in self.voxel_files:
            voxel_path = self.voxel_files[seq_name]
            try:
                # Load voxel data
                voxel_data = np.load(voxel_path)
                
                # Try to get the voxel data from the file
                if 'voxel' in voxel_data:
                    voxel_tensor = torch.from_numpy(voxel_data['voxel']).float()
                elif 'event_representation' in voxel_data:
                    voxel_tensor = torch.from_numpy(voxel_data['event_representation']).float()
                else:
                    # Get the first array in the file
                    key = list(voxel_data.keys())[0]
                    voxel_tensor = torch.from_numpy(voxel_data[key]).float()
                
                # Normalize voxel data
                voxel_tensor = (voxel_tensor - voxel_tensor.mean()) / (voxel_tensor.std() + 1e-6)
                
                # Make sure the voxel tensor has the right shape (C, H, W)
                if voxel_tensor.dim() == 3 and voxel_tensor.shape[0] != 6:
                    # If channels not in first dimension, rearrange
                    if voxel_tensor.shape[2] == 6:
                        voxel_tensor = voxel_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
                # Resize voxel data to match the blur image if needed
                if voxel_tensor.shape[1] != blur_img.shape[1] or voxel_tensor.shape[2] != blur_img.shape[2]:
                    voxel_tensor = torch.nn.functional.interpolate(
                        voxel_tensor.unsqueeze(0),
                        size=(blur_img.shape[1], blur_img.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
            except Exception as e:
                print(f"Error loading voxel file {voxel_path}: {e}")
                voxel_tensor = None
        
        # If no valid voxel tensor was loaded, create a dummy one
        if voxel_tensor is None:
            print(f"WARNING: Using dummy voxel tensor for {seq_name}")
            voxel_tensor = torch.zeros((6, blur_img.shape[1], blur_img.shape[2])).float()
        
        return {
            'lq': blur_img,
            'event': voxel_tensor,
            'seq_name': seq_name,
            'lq_path': blur_path
        }


def main():
    # Parse options, set distributed setting, set random seed
    opt = parse_options(is_train=False)
    torch.backends.cudnn.benchmark = True
    
    # Check and prepare output directory
    output_dir = os.path.join(opt['path']['results_root'], 'visualization', opt['name'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    log_file = osp.join(opt['path']['log'],
                        f"test_output_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    # Load checkpoint path from options
    checkpoint_path = opt['path'].get('pretrain_network_g', None)
    if not checkpoint_path or not osp.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Create test dataset
    test_dataset = TestDataset(
        dataroot=opt['datasets']['test']['dataroot'],
        dataroot_voxel=opt['datasets']['test']['dataroot_voxel']
    )
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=1  # Reduced to help with debugging
    )
    
    # Create model
    model = create_model(opt)
    
    # Start testing
    logger.info(f'Testing {len(test_dataset)} images...')
    model.net_g.eval()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            try:
                # Move data to device
                lq = data['lq'].to(model.device)
                event = data['event'].to(model.device)
                seq_name = data['seq_name'][0]
                
                # Verify shapes
                logger.info(f"Processing {seq_name}: blur shape {lq.shape}, voxel shape {event.shape}")
                
                # Run model
                output = model.net_g(lq, event)
                
                # Save output image
                output_img = output[0].cpu().numpy().transpose(1, 2, 0)
                output_img = (output_img * 255).clip(0, 255).astype(np.uint8)
                output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(output_dir, f"{seq_name}_deblurred.png")
                cv2.imwrite(output_path, output_img)
                
                logger.info(f"Saved output to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {data['seq_name']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info(f"Testing completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()