import torch
from torch import nn

class CouplingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels//2 * 2, 3, padding=1)  # outputs scale & shift
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], dim=1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=1)

class PyramidAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels//8, 1)
        self.k = nn.Conv2d(channels, channels//8, 1)
        self.v = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, -1, h*w)           # b x c' x N
        k = self.k(x).view(b, -1, h*w)           # b x c' x N
        v = self.v(x).view(b, c,   h*w)          # b x c  x N

        attn = torch.softmax(q.transpose(1,2) @ k / (c**0.5), dim=-1)  # b x N x N
        out = (v @ attn).view(b, c, h, w)
        return out + x  # residual
