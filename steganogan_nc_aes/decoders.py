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

    def __init__(self, data_depth, password_depth):
        super(BasicDecoder, self).__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.password_depth = password_depth
        self._build_layers()

    def _build_layers(self):
        # Decoder layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer1_bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, self.data_depth + self.password_depth, kernel_size=3, padding=1)

        # Cosine similarity processor for smooth gating
        self.similarity_processor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.data_depth, kernel_size=3, padding=1),
            nn.Sigmoid()  # Smooth gating weights [0,1]
        )

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x, p):
        # First block
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)

        # Second layer
        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)

        # Third layer
        x3 = self.conv3(x2)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)

        # Output layer
        raw_data = self.conv4(x3)

        decoded_message = raw_data[:, :self.data_depth, :, :] # (N, D, H, W)
        decoded_password = raw_data[:, self.data_depth:, :, :] # (N, P, H, W)
        
        # Calculate cosine similarity (naturally bounded [-1,1])
        decoded_flat = decoded_password.view(decoded_password.shape[0], -1) # (N, P*H*W)
        p_flat = p.view(p.shape[0], -1) # (N, P*H*W)
        
        cosine_sim = F.cosine_similarity(decoded_flat, p_flat, dim=1)
        cosine_sim = (cosine_sim + 1) / 2  # Scale from [-1,1] to [0,1]
        
        # Expand similarity to spatial dimensions
        similarity_map = cosine_sim.view(-1, 1, 1, 1).expand(-1, 1, decoded_message.shape[2], decoded_message.shape[3]) # (N, 1, H, W)
        
        # Process similarity into gating weights
        gate_weights = self.similarity_processor(similarity_map) # (N, D, H, W)
        
        # Apply gating: message * gate_weights
        # When similarity is low (passwords don't match), gate_weights approach 0
        # When similarity is high (passwords match), gate_weights approach 1
        # This naturally returns zeros for wrong passwords and message for correct ones
        gated_message = decoded_message * gate_weights # (N, D, H, W)
        
        return gated_message


class DenseDecoder(nn.Module):
    """
    Decoder using cosine similarity for smooth password verification
    Returns zeros when passwords don't match
    """
    def __init__(self, data_depth, password_depth):
        super(DenseDecoder, self).__init__()
        self.version = '4'
        self.data_depth = data_depth
        self.password_depth = password_depth
        self._build_layers()
        
    def _build_layers(self):
        # Base decoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(96, self.data_depth + self.password_depth, kernel_size=3, padding=1)
        
        # Cosine similarity processor for smooth gating
        self.similarity_processor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.data_depth, kernel_size=3, padding=1),
            nn.Sigmoid()  # Smooth gating weights [0,1]
        )

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '1'

    def forward(self, x, p):
        # Base decoding
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)
        
        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)
        
        x3_input = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3_input)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)
        
        x4_input = torch.cat([x1, x2, x3], dim=1)
        raw_data = self.conv4(x4_input)
        
        decoded_message = raw_data[:, :self.data_depth, :, :] # (N, D, H, W)
        decoded_password = raw_data[:, self.data_depth:, :, :] # (N, P, H, W)
        
        # Calculate cosine similarity (naturally bounded [-1,1])
        decoded_flat = decoded_password.view(decoded_password.shape[0], -1) # (N, P*H*W)
        p_flat = p.view(p.shape[0], -1) # (N, P*H*W)
        
        cosine_sim = F.cosine_similarity(decoded_flat, p_flat, dim=1)
        cosine_sim = (cosine_sim + 1) / 2  # Scale from [-1,1] to [0,1]
        
        # Expand similarity to spatial dimensions
        similarity_map = cosine_sim.view(-1, 1, 1, 1).expand(-1, 1, decoded_message.shape[2], decoded_message.shape[3]) # (N, 1, H, W)
        
        # Process similarity into gating weights
        gate_weights = self.similarity_processor(similarity_map) # (N, D, H, W)
        
        # Apply gating: message * gate_weights
        # When similarity is low (passwords don't match), gate_weights approach 0
        # When similarity is high (passwords match), gate_weights approach 1
        # This naturally returns zeros for wrong passwords and message for correct ones
        gated_message = decoded_message * gate_weights # (N, D, H, W)
        
        return gated_message