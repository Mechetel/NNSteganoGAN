import torch
from torch import nn
import torch.nn.functional as F
from .model_utils import ResidualBlock, PyramidAttention


class SteganoEncoder(nn.Module):
    """
    PANet-based Steganography Encoder
    Embeds secret data into cover images
    """
    def __init__(self, secret_channels=3, cover_channels=3):
        self.hidden_channels = 64
        super(SteganoEncoder, self).__init__()

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(secret_channels + 3, hidden_channels, 3, padding=1)

        # Encoder blocks with pyramid attention
        self.encoder_blocks = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels * 4),
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
        )

        # Middle pyramid attention block
        self.middle_attention = PyramidAttention(hidden_channels * 4)

        # Decoder blocks
        self.decoder_blocks = nn.Sequential(
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
            ResidualBlock(hidden_channels * 4, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
        )

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, secret, cover):
        # Concatenate secret and cover images
        x = torch.cat([secret, cover], dim=1)

        # Initial feature extraction
        x = F.relu(self.initial_conv(x))

        # Encode with pyramid attention
        x = self.encoder_blocks(x)

        # Middle attention refinement
        x = self.middle_attention(x)

        # Decode to stego image
        x = self.decoder_blocks(x)

        # Final stego image
        stego = self.final_conv(x)

        return stego
