import torch
import torch.nn.functional as F
import math
import os
import glob
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride = stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super(ConvLayerNorm, self).__init__()
        # LayerNorm will normalize over the last dimension (channels after permutation)
        self.layer_norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        # x has shape (N, C, H, W); we permute to (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            ConvLayerNorm(out_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_ch, out_ch),
            ConvLayerNorm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        skip_input = x
        output = self.pool(x)
        return skip_input, output

class Up(nn.Module):
    # Upsamples the input from bottleneck and concatenates the skip connection.
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
    def forward(self, skip, x):
        x = self.up_scale(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return x

class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(out_ch*2, out_ch)

    def forward(self, skip_connection, input):
        x = self.up(skip_connection, input)
        x = self.conv(x)
        return x

class PatchTokenizer(nn.Module):
    """
    Converts a feature map of shape (B, in_channels, H, W) into
    a sequence of tokens of shape (B, N, token_dim), where N = (H/patch_size) * (W/patch_size).
    """
    def __init__(self, in_channels, patch_size, token_dim):
        super(PatchTokenizer, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        x = self.proj(x)
        B, token_dim, H_p, W_p = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens

class TokenSelfAttention(nn.Module):
    def __init__(self, token_dim, num_heads, dropout=0.1):
        super(TokenSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4),
            nn.ReLU(),
            nn.Linear(token_dim * 4, token_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # tokens shape: (B, N, token_dim)
        # Transpose to (N, B, token_dim) as required by nn.MultiheadAttention.
        tokens_trans = tokens.transpose(0, 1)
        
        # Self-attention: query, key, and value are the same.
        attn_output, _ = self.attention(tokens_trans, tokens_trans, tokens_trans)
        # Add & Norm
        tokens_attn = self.norm1(tokens_trans + self.dropout(attn_output))
        
        # Feed-forward network with residual connection.
        ff_output = self.ff(tokens_attn)
        tokens_ff = self.norm2(tokens_attn + self.dropout(ff_output))
        
        # Transpose back to (B, N, token_dim)
        return tokens_ff.transpose(0, 1)


#############################################
# NEW: An Event Encoder to process event data
#############################################

class EventEncoder(nn.Module):
    """
    A simple encoder for event data.
    Assumes the event data is provided as a tensor of shape (B, C_events, H, W)
    where C_events is (for example) 10.
    The encoder downsamples the spatial resolution by a factor of 16 (4 conv layers with stride 2)
    and outputs a feature map with 1024 channels (to match the bottleneck of the image branch).
    """
    def __init__(self, in_channels=10, out_channels=1024):
        super(EventEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Final conv to get desired channel count
        self.conv5 = DepthwiseSeparableConv(512, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.conv2(x)  # (B, 128, H/4, W/4)
        x = self.conv3(x)  # (B, 256, H/8, W/8)
        x = self.conv4(x)  # (B, 512, H/16, W/16)
        x = self.conv5(x)  # (B, 1024, H/16, W/16)
        return x
    
class KUnet(nn.Module):
    def __init__(self, dimensions=3, patch_size=2, token_dim=1024, tokenization=True, num_heads = 8, event_input_channels = 6):
        super(KUnet, self).__init__()
        self.tokenization = tokenization
        self.down1 = DownLayer(3, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256, 512)

        # fuse events with images. 
        self.bottleneck = DoubleConv(512, 1024)
        self.event_encoder = EventEncoder(in_channels= event_input_channels, out_channels= token_dim)
        self.fusion_layer = nn.Conv2d(2048,1024,kernel_size = 1)

    
        self.tokenizer = PatchTokenizer(1024, patch_size, token_dim)
        self.self_attention = TokenSelfAttention(token_dim=token_dim, num_heads=num_heads)


        self.token_projection1 = KANLinear(token_dim,2048)
        self.token_projection2 = KANLinear(2048,3072)
        self.token_projection3 = KANLinear(3072, 1024)
        
        
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        self.last_conv = nn.Conv2d(64, dimensions, 1)
        
        # #Explicitly define normalization parameters
        # self.register_buffer('norm_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        # self.register_buffer('norm_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def preprocess_input(self, x):
        """Normalize input tensor from [0, 1] to [-1, 1]."""
        return (x - 0.5) / 0.5

    def postprocess(self, x):
        """Denormalize tensor from [-1, 1] back to [0, 1]."""
        return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


    def forward(self, blurred_images, event_data):

        x = self.preprocess_input(blurred_images)
        # Encoder
        x1_skip, x1 = self.down1(x)
        x2_skip, x2 = self.down2(x1)
        x3_skip, x3 = self.down3(x2)
        x4_skip, x4 = self.down4(x3)

        # Bottleneck process event features and fuse with bottleneck
        x5 = self.bottleneck(x4)
        event_features = self.event_encoder(event_data)
        x5 = torch.cat([x5,event_features], dim = 1)
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