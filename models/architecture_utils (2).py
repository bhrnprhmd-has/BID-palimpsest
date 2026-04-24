import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # You may need to: pip install einops

class TransformerBottleneck(nn.Module):
    """
    Applies Multi-Head Self-Attention at the bottleneck of the network.
    This gives the model 'Global Context' to connect widely separated ink strokes.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.num_heads = num_heads
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-Head Self Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward Network (MLP)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x is a spatial feature map from the CNN: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        
        # 1. Flatten spatial dimensions to create a "Sequence of Patches"
        # Shape becomes: (Batch, Sequence_Length, Channels) where Sequence_Length = H * W
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # 2. Apply Self-Attention (with Residual Connection)
        # We attend the sequence to itself
        attn_input = self.norm1(x_flat)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x_flat = x_flat + attn_out
        
        # 3. Apply MLP (with Residual Connection)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # 4. Reshape back to Spatial Image Map
        out = rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Helps the model focus on 'faint' signals by re-weighting important channels and spatial regions.
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 1. Channel Attention
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # 2. Spatial Attention
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
        
        # 1. Attention Gate (Focus on the faint ink)
        self.attention = CBAM(input_nc)

        # 2. Context Aggregation (Dilated Blocks)
        # Dilations: 1, 2, 4, 8 -> Sees very large context
        self.context_stream = nn.Sequential(
            DilatedResBlock(input_nc, dilation=1),
            DilatedResBlock(input_nc, dilation=2),
            DilatedResBlock(input_nc, dilation=4),
            DilatedResBlock(input_nc, dilation=8)
        )

        # 3. Upsampling Stream (Standard decoding)
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