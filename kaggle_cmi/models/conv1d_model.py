from typing import List, Any

import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

from kaggle_cmi.models.model import SequenceClassifierPyTorch


class Conv1DClassifier(nn.Module):
    def __init__(
        self,
        feature_cols: List[str],
        n_classes: int,
        label_encoder: Any = LabelEncoder(),
        scaler: Any = StandardScaler(),
        **kwargs,
    ):
        super().__init__()
        self.feature_cols = feature_cols
        self.label_encoder = label_encoder
        self.scaler = scaler

        self.conv1 = nn.Conv1d(
            in_channels=len(self.feature_cols),
            out_channels=64,
            kernel_size=7,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm1d(512)

        # Pooling and dropout
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Conv1DSequenceClassifier(SequenceClassifierPyTorch):
    name = "conv1d"
    model_class = Conv1DClassifier
