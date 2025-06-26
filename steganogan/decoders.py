# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class BasicDecoder(nn.Module):
    """
    The BasicDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth
        
        # Define individual layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, self.data_depth, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass through the decoder network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)
        
        x = self.conv4(x)
        return x


class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor with dense connections.
    Input: (N, 3, H, W)
    Output: (N, D, H, W)
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth
        
        # Define layers with proper input channels for dense connections
        # conv1: input channels = 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # conv2: input channels = 32 (from conv1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # conv3: input channels = 64 (conv1 + conv2 concatenated: 32 + 32)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # conv4: input channels = 96 (conv1 + conv2 + conv3 concatenated: 32 + 32 + 32)
        self.conv4 = nn.Conv2d(96, self.data_depth, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass with dense connections - concatenating all previous outputs."""
        # First layer
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, inplace=True)
        
        # Second layer - input is just x1
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, inplace=True)
        
        # Third layer - input is concatenation of x1 and x2
        x3_input = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension
        x3 = self.conv3(x3_input)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, inplace=True)
        
        # Fourth layer - input is concatenation of x1, x2, and x3
        x4_input = torch.cat([x1, x2, x3], dim=1)  # Concatenate all previous outputs
        x4 = self.conv4(x4_input)
        
        return x4