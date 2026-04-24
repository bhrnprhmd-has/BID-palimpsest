import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # You may need to: pip install einops

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Helps the model focus on 'faint' signals by re-weighting important channels and spatial regions.
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1), # Helps stabilize gradients
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        # MaxPool + AvgPool -> MLP
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        scale = self.channel_sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * scale

        # Spatial Attention
        # Max + Avg across channels -> Conv2d
        compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(compress)
        return x * spatial_scale

class DilatedResBlock(nn.Module):
    """Residual Block with Dilation to increase Receptive Field without losing resolution."""
    def __init__(self, dim, dilation=1):
        super(DilatedResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ContextAwareDecoder(nn.Module):
    """
    Specialized Decoder for Undertext (netH2).
    Combines Attention (to find faint text) + Dilation (to connect broken strokes).
    """
    def __init__(self, input_nc, output_nc, ngf=64):
        super(ContextAwareDecoder, self).__init__()
        
        # Attention Gate (Focus on the faint ink)
        self.attention = CBAM(input_nc)

        # Context Aggregation (Dilated Blocks)
        # Dilations: 1, 2, 4, 8 -> Sees very large context
        self.context_stream = nn.Sequential(
            DilatedResBlock(input_nc, dilation=1),
            DilatedResBlock(input_nc, dilation=2),
            DilatedResBlock(input_nc, dilation=4),
            DilatedResBlock(input_nc, dilation=8)
        )

        # Upsampling Stream (Standard decoding)
        model = []
        # Upsample 1 (Assuming input was downsampled 4x total)
        model += [nn.ConvTranspose2d(input_nc, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True)]
        # Upsample 2
        model += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True)]
        # Final Output
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.upsample_stream = nn.Sequential(*model)

    def forward(self, x):
        x = self.attention(x)    # Step 1: Highlight faint text
        x = self.context_stream(x) # Step 2: Connect broken strokes
        return self.upsample_stream(x) # Step 3: Generate image
