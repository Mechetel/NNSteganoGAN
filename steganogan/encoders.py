# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    def __init__(self, data_depth):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth

        # Feature extraction layers
        self.feat_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feat_bn = nn.BatchNorm2d(32)

        # Main processing layers
        self.conv1 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)


    def forward(self, image, data):
        # Feature extraction from image
        x = self.feat_conv(image)
        x = F.leaky_relu(x, inplace=True)
        x = self.feat_bn(x)

        # Concatenate features with data and process
        x = torch.cat([x, data], dim=1)
        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.tanh(x)

        return x


class ResidualEncoder(nn.Module):
    """
    The ResidualEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth

        # Feature extraction layers
        self.feat_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feat_bn = nn.BatchNorm2d(32)

        # Main processing layers
        self.conv1 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)


    def forward(self, image, data):
        # Feature extraction from image
        x = self.feat_conv(image)
        x = F.leaky_relu(x, inplace=True)
        x = self.feat_bn(x)

        # Concatenate features with data and process
        x = torch.cat([x, data], dim=1)
        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn2(x)

        x = self.conv3(x)
        # No activation on final layer

        x = image + x
        return x


class DenseEncoder(nn.Module):
    """
    The DenseEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image with dense connections.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth

        # First layer: image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second layer: features + data
        self.conv2 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third layer: previous features + data (dense connection)
        self.conv3 = nn.Conv2d(64 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fourth layer: all previous features + data (dense connection)
        self.conv4 = nn.Conv2d(96 + self.data_depth, 3, kernel_size=3, padding=1)

    def forward(self, image, data):
        # First layer: extract image features
        x1 = self.conv1(image)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)

        # Second layer: combine x1 with data
        x2_input = torch.cat([x1, data], dim=1)
        x2 = self.conv2(x2_input)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)

        # Third layer: combine x1, x2 with data (dense connection)
        x3_input = torch.cat([x1, x2, data], dim=1)
        x3 = self.conv3(x3_input)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)

        # Fourth layer: combine x1, x2, x3 with data (dense connection)
        x4_input = torch.cat([x1, x2, x3, data], dim=1)
        x4 = self.conv4(x4_input)
        # No activation on final layer

        x4 = image + x4
        return x4