import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualSECNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        pool_size: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second conv block
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding="same", bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        # SE block
        out = self.se(out)

        # Add shortcut
        out += shortcut
        out = F.relu(out)

        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)

        return out


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context
