import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.models.archs.KUnet import KUnet, EventEncoder, DepthwiseSeparableConv

# Include the adapter code directly to avoid import issues
class EventEncoder12To6Adapter(torch.nn.Module):
    """Adapter that takes 12-bin voxel input but works with a 6-bin trained EventEncoder"""
    def __init__(self, original_encoder):
        super(EventEncoder12To6Adapter, self).__init__()
        self.original_encoder = original_encoder
        # Create a 1x1 conv to reduce 12 channels to 6
        self.adapter = torch.nn.Conv2d(12, 6, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # x has shape [B, 12, H, W]
        x = self.adapter(x)  # Convert to [B, 6, H, W]
        return self.original_encoder(x)

class KUnet12BinAdapter(torch.nn.Module):
    """Wrapper around KUnet that allows 12-bin input to a 6-bin trained model"""
    def __init__(self, original_model):
        super(KUnet12BinAdapter, self).__init__()
        # Keep all original components
        self.tokenization = original_model.tokenization
        self.down1 = original_model.down1
        self.down2 = original_model.down2
        self.down3 = original_model.down3
        self.down4 = original_model.down4
        self.bottleneck = original_model.bottleneck
        self.fusion_layer = original_model.fusion_layer
        self.tokenizer = original_model.tokenizer
        self.self_attention = original_model.self_attention
        self.token_projection1 = original_model.token_projection1
        self.token_projection2 = original_model.token_projection2
        self.token_projection3 = original_model.token_projection3
        self.up1 = original_model.up1
        self.up2 = original_model.up2
        self.up3 = original_model.up3
        self.up4 = original_model.up4
        self.last_conv = original_model.last_conv
        
        # Replace the event encoder with our adapter
        self.event_encoder = EventEncoder12To6Adapter(original_model.event_encoder)
    
    def preprocess_input(self, x):
        return (x - 0.5) / 0.5
        
    def postprocess(self, x):
        return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
    
    def forward(self, blurred_images, event_data):
        # Same forward pass as original KUnet
        x = self.preprocess_input(blurred_images)
        # Encoder
        x1_skip, x1 = self.down1(x)
        x2_skip, x2 = self.down2(x1)
        x3_skip, x3 = self.down3(x2)
        x4_skip, x4 = self.down4(x3)

        # Bottleneck process event features and fuse with bottleneck
        x5 = self.bottleneck(x4)
        event_features = self.event_encoder(event_data)  # This now handles 12-bin input
        x5 = torch.cat([x5, event_features], dim=1)
        x5 = self.fusion_layer(x5)

        # Optional tokenization
        if self.tokenization: 
            tokens = self.tokenizer(x5)
            tokens = self.self_attention(tokens)
            tokens = self.token_projection1(tokens)
            tokens = self.token_projection2(tokens)
            tokens = self.token_projection3(tokens)

            B, N, _ = tokens.shape
            H_b = x5.shape[2] // self.tokenizer.patch_size
            W_b = x5.shape[3] // self.tokenizer.patch_size
            x5_reconstructed = tokens.transpose(1, 2).reshape(B, 1024, H_b, W_b)
            x = self.up1(x4_skip, x5_reconstructed)
        else:
            x = self.up1(x4_skip, x5)

        # Decoder
        x = self.up2(x3_skip, x)
        x = self.up3(x2_skip, x)
        x = self.up4(x1_skip, x)
        x = self.last_conv(x)
        x = self.postprocess(x)
        return x

def load_model_for_12bin_testing(model_path):
    """Load a 6-bin trained model and adapt it for 12-bin testing"""
    # First load the original model with 6 bins
    model = KUnet(event_input_channels=6)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint structures
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)
    
    # Now wrap it with our adapter
    model_12bin = KUnet12BinAdapter(model)
    return model_12bin

def main():
    # parse options, set distributed setting, set random seed
    opt = parse_options(is_train=False)

    # Print path section for debugging
    print("Path section:", opt['path'])
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Get model path
    model_path = opt['path'].get('pretrain_network_g')
    if model_path is None:
        logger.error('No pretrained model found. Please specify it in the config file.')
        return
    
    logger.info(f'Loading pretrained model from {model_path}')
    
    # Always use our 12-bin adapter for this script
    logger.info('Testing with 12-bin voxel data using adapter')
    
    # Load model with adapter first, before creating the testing model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapted_model = load_model_for_12bin_testing(model_path)
    adapted_model = adapted_model.to(device)
    
    # Make a copy of the options and remove the pretrain path so it won't try to load the weights
    test_opt = opt.copy()
    if 'pretrain_network_g' in test_opt['path']:
        test_opt['path']['pretrain_network_g'] = None
    
    # Create the model with testing options (without loading weights)
    model = create_model(test_opt)
    
    # Replace the network with our adapted model that already has the weights loaded
    model.net_g = adapted_model
    logger.info('Model successfully adapted for 12-bin testing')

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    for test_loader in test_loaders:
        test_set_name = opt['datasets']['test']['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # whether to use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)


if __name__ == '__main__':
    main() 