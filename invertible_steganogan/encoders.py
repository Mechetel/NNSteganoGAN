import torch
from torch import nn
import torch.nn.functional as F
from .model_utils import CouplingBlock, PyramidAttention


class InvertibleStegaEncoder(nn.Module):
    """
    Простая одномасштабная инвертируемая модель без изменения разрешения
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth
        self.base_channels = 64
        self.n_blocks = 4
        
        ch = self.base_channels
        # начальная свёртка: cover(3) + data(data_depth) -> ch каналов
        self.initial = nn.Sequential(
            nn.Conv2d(3 + self.data_depth, ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # повторяющиеся блоки attention + coupling
        blocks = []
        for _ in range(self.n_blocks):
            blocks.append(PyramidAttention(ch))
            blocks.append(CouplingBlock(ch))
        self.blocks = nn.Sequential(*blocks)
        # финальный свёрточный слой: ch -> RGB
        self.final = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, cover, data):
        # data: тензор размера (N, data_depth, H, W)
        x = torch.cat([cover, data], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        stego = torch.tanh(self.final(x))  # (N,3,H,W)
        return stego
