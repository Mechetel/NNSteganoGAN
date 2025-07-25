import torch
from torch import nn
import torch.nn.functional as F
from .model_utils import CouplingBlock, PyramidAttention


class InvertibleStegaDecoder(nn.Module):
    """
    Декодер для одноразрешенной модели
    """
    def __init__(self, data_depth):
        super().__init__()
        self.data_depth = data_depth
        self.base_channels = 64
        self.n_blocks = 4

        ch = self.base_channels
        # начальная свёртка: stego(3) -> ch каналов
        self.initial = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # обратные блоки: coupling + attention
        blocks = []
        for _ in range(self.n_blocks):
            blocks.append(CouplingBlock(ch))
            blocks.append(PyramidAttention(ch))
        self.rev_blocks = nn.Sequential(*blocks)
        # финальный слой: ch -> data_depth
        self.output = nn.Conv2d(ch, self.data_depth, 3, padding=1)

    def forward(self, stego):
        x = self.initial(stego)
        # инвертируем coupling-блоки в обратном порядке
        for layer in reversed(self.rev_blocks):
            if isinstance(layer, CouplingBlock):
                x = layer(x, reverse=True)
            else:
                x = layer(x)
        data_hat = torch.sigmoid(self.output(x))  # (N,data_depth,H,W)
        return data_hat
