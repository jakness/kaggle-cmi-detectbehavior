from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

from kaggle_cmi.models.model import SequenceClassifierPyTorch
from kaggle_cmi.models.torch_nn_utils import ResidualSECNNBlock, AttentionLayer


class TwoBranchClassifier(nn.Module):
    def __init__(
        self,
        imu_feature_cols: List[str],
        thm_tof_feature_cols: List[str],
        feature_cols: List[str],
        n_classes: int,
        label_encoder: Any = LabelEncoder(),
        scaler: Any = StandardScaler(),
        **kwargs,
    ):
        if len(thm_tof_feature_cols) == 0:
            raise ValueError("'thm_tof_feature_cols' must not be empty")
        super().__init__()
        self.imu_feature_cols = imu_feature_cols
        self.thm_tof_feature_cols = thm_tof_feature_cols
        self.feature_cols = feature_cols
        self.imu_indices_in_feature_cols = [
            feature_cols.index(col) for col in imu_feature_cols
        ]
        self.thm_tof_indices_in_feature_cols = [
            feature_cols.index(col) for col in thm_tof_feature_cols
        ]
        self.label_encoder = label_encoder
        self.scaler = scaler

        # imu branch
        self.imu_block1 = ResidualSECNNBlock(
            in_channels=len(imu_feature_cols),
            out_channels=64,
            kernel_size=3,
            dropout=0.3,
        )
        self.imu_block2 = ResidualSECNNBlock(
            in_channels=64, out_channels=128, kernel_size=5, dropout=0.3
        )

        # thm/tof branch
        self.tof_conv1 = nn.Conv1d(
            in_channels=len(thm_tof_feature_cols),
            out_channels=64,
            kernel_size=3,
            padding="same",
            bias=False,
        )
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(0.3)

        self.tof_conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same", bias=False
        )
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(0.3)

        self.bilstm = nn.LSTM(
            input_size=256, hidden_size=128, bidirectional=True, batch_first=True
        )
        self.lstm_dropout = nn.Dropout(0.4)

        self.attention = AttentionLayer(256)

        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        imu = x[:, self.imu_indices_in_feature_cols, :]
        thm_tof = x[:, self.thm_tof_indices_in_feature_cols, :]

        # imu branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)

        # thm/tof branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(thm_tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))

        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)

        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)

        attended = self.attention(lstm_out)

        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)

        x = self.classifier(x)
        return x


class TwoBranchSequenceClassifier(SequenceClassifierPyTorch):
    name = "two_branch"
    model_class = TwoBranchClassifier
