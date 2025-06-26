import torch
from torch import nn
import torch.nn.functional as F

class BasicCritic(nn.Module):
    """
    The BasicCritic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).
    Input: (N, 3, H, W)
    Output: (N, 1)
    """
    def __init__(self):
        super().__init__()
        self.version = '1'
        
        # Define individual layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3)

    def forward(self, x):
        """Forward pass through the critic network."""
        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.bn3(x)
        
        x = self.conv4(x)  # (N, 1, H', W')
        
        # Global average pooling: reduce spatial dimensions to single value
        x = torch.mean(x.view(x.size(0), -1), dim=1)  # (N, 1)
        return x