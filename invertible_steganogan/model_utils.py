import torch
from torch import nn

class PyramidAttention(nn.Module):
    """
    Pyramid Attention Module inspired by PANet
    Captures long-range feature correspondences from multi-scale feature pyramid
    """
    def __init__(self, in_channels, reduction=16):
        super(PyramidAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Multi-scale feature extraction
        self.pyramid_conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.pyramid_conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.pyramid_conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=False)

        # Feature fusion
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)

    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.pyramid_conv1(x)
        feat3 = self.pyramid_conv3(x)
        feat5 = self.pyramid_conv5(x)

        # Feature concatenation
        multi_scale_feat = torch.cat([feat1, feat3, feat5], dim=1)
        fused_feat = self.fusion_conv(multi_scale_feat)

        # Channel attention
        channel_att = self.channel_attention(fused_feat)
        channel_refined = fused_feat * channel_att

        # Spatial attention
        avg_out = torch.mean(channel_refined, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        # Final attention-refined features
        output = channel_refined * spatial_att

        return output + x  # Residual connection

class ResidualBlock(nn.Module):
    """Residual block with pyramid attention"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pyramid_attention = PyramidAttention(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pyramid_attention(out)

        out += residual
        out = F.relu(out)

        return out
