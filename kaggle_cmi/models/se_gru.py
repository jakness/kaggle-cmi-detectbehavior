from typing import List, Any

import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

from kaggle_cmi.models.torch_nn_utils import ResidualSECNNBlock, AttentionLayer
from kaggle_cmi.models.model import SequenceClassifierPyTorch


class SqueezeExcitationGRUClassifier(nn.Module):
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

        self.imu_block1 = ResidualSECNNBlock(
            in_channels=len(feature_cols), out_channels=64, kernel_size=3, dropout=0.3
        )
        self.imu_block2 = ResidualSECNNBlock(
            in_channels=64, out_channels=128, kernel_size=5, dropout=0.3
        )

        self.bi_gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)

        self.attention = AttentionLayer(256)

        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.imu_block1(x)
        x = self.imu_block2(x)
        x = x.transpose(1, 2)

        gru_out, _ = self.bi_gru(x)  # (B, T'', 2H)
        gru_out = self.gru_dropout(gru_out)

        attended = self.attention(gru_out)  # (B, 2H)

        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        x = self.classifier(x)
        return x


class SqueezeExcitationGRUSequenceClassifier(SequenceClassifierPyTorch):
    name = "se_gru"
    model_class = SqueezeExcitationGRUClassifier
