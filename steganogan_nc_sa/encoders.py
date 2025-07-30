# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import math


class SelfAttention2D(nn.Module):
    """
    Self-attention block для 2D изображений
    """
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Уменьшаем размерность для эффективности
        self.hidden_dim = max(in_channels // reduction, 1)
        
        # Query, Key, Value проекции
        self.query_conv = nn.Conv2d(in_channels, self.hidden_dim, 1)
        self.key_conv = nn.Conv2d(in_channels, self.hidden_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Выходная проекция
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # Параметр для постепенного включения attention
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Нормализация
        self.layer_norm = nn.GroupNorm(1, in_channels)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Сохраняем оригинальный вход для residual connection
        residual = x
        
        # Применяем нормализацию
        x = self.layer_norm(x)
        
        # Вычисляем Q, K, V
        query = self.query_conv(x).view(batch_size, self.hidden_dim, -1)  # B x C' x N
        key = self.key_conv(x).view(batch_size, self.hidden_dim, -1)      # B x C' x N  
        value = self.value_conv(x).view(batch_size, channels, -1)         # B x C x N
        
        # Транспонируем для матричного умножения
        query = query.permute(0, 2, 1)  # B x N x C'
        
        # Вычисляем attention weights
        attention_weights = torch.bmm(query, key)  # B x N x N
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.hidden_dim, device=x.device))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Применяем attention к values
        value = value.permute(0, 2, 1)  # B x N x C
        attended = torch.bmm(attention_weights, value)  # B x N x C
        attended = attended.permute(0, 2, 1)  # B x C x N
        
        # Возвращаем к исходной форме
        attended = attended.view(batch_size, channels, height, width)
        
        # Применяем выходную проекцию
        attended = self.out_conv(attended)
        
        # Residual connection с learnable scaling
        output = residual + self.gamma * attended
        
        return output


class DenseEncoderWithAttention(nn.Module):
    """
    The DenseEncoder with Self-Attention module takes a cover image and a data tensor 
    and combines them into a steganographic image with dense connections and attention mechanisms.
    Input: (N, 3, H, W), (N, D, H, W)
    Output: (N, 3, H, W)
    """

    def __init__(self, data_depth):
        super(DenseEncoderWithAttention, self).__init__()
        self.version = '2'
        self.data_depth = data_depth
        self._build_layers()

    def _build_layers(self):
        # First convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Self-attention после первой свертки (ранние признаки)
        self.attention1 = SelfAttention2D(32, reduction=8)

        # Second convolution block
        self.conv2 = nn.Conv2d(32 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution block (dense connection)
        self.conv3 = nn.Conv2d(64 + self.data_depth, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Self-attention перед финальным выводом (поздние признаки)
        self.attention3 = SelfAttention2D(32, reduction=8)

        # Output convolution
        self.conv4 = nn.Conv2d(96 + self.data_depth, 3, kernel_size=3, padding=1)

    def upgrade_legacy(self):
        """Transform legacy pretrained models to make them usable with new code versions."""
        if not hasattr(self, 'version'):
            self.version = '2'

    def forward(self, image, data):
        # First block с attention
        x1 = self.conv1(image)
        x1 = F.leaky_relu(x1, inplace=True)
        x1 = self.bn1(x1)
        
        # Применяем self-attention для захвата ранних пространственных паттернов
        x1 = self.attention1(x1)

        # Second block с dense connection
        x2_input = torch.cat([x1, data], dim=1)
        x2 = self.conv2(x2_input)
        x2 = F.leaky_relu(x2, inplace=True)
        x2 = self.bn2(x2)

        # Third block с dense connections
        x3_input = torch.cat([x1, x2, data], dim=1)
        x3 = self.conv3(x3_input)
        x3 = F.leaky_relu(x3, inplace=True)
        x3 = self.bn3(x3)
        
        # Применяем self-attention для глобального согласования признаков
        x3 = self.attention3(x3)

        # Output block со всеми dense connections
        x4_input = torch.cat([x1, x2, x3, data], dim=1)
        output = self.conv4(x4_input)

        # Add residual connection (original image)
        output = image + output

        return output