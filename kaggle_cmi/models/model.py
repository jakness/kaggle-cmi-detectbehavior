import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Type, Union
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kaggle_cmi.data.utils import SequenceClassifierDataset


logger = logging.getLogger(__name__)


MODEL_PATH_SUFFIX = ".pkl"
MODEL_WEIGHTS_PATH_SUFFIX = "_model_weights.pth"


class Model(ABC):
    name: str
    model_class: Type[nn.Module]

    def __init__(
        self,
        feature_cols: List[str],
        predicted_col: str,
        n_classes: int,
        sequence_length: int,
        **kwargs,
    ):
        self.feature_cols = feature_cols
        self.predicted_col = predicted_col
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.model = self.model_class(
            feature_cols=self.feature_cols, n_classes=self.n_classes, **kwargs
        )

    @abstractmethod
    def train_model(self, **kwargs):
        pass

    @abstractmethod
    def save_model(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, restore_best_weights: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class SequenceClassifierPyTorch(Model):
    def train_model(
        self,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame,
        sequence_id_col: str,
        n_epochs: int,
        use_early_stopping: bool = True,
    ):
        logger.info("Creating datasets and dataloaders for training and validation")
        train_data = train_data.with_columns(
            pl.Series(
                self.predicted_col,
                self.model.label_encoder.fit_transform(train_data[self.predicted_col]),
            )
        )
        validation_data = validation_data.with_columns(
            pl.Series(
                self.predicted_col,
                self.model.label_encoder.transform(validation_data[self.predicted_col]),
            )
        )

        train_data = self.scale_features(df=train_data, fit_scaler=True)
        validation_data = self.scale_features(df=validation_data, fit_scaler=True)

        train_dataset = SequenceClassifierDataset(
            df=train_data,
            sequence_id_col=sequence_id_col,
            feature_cols=self.feature_cols,
            label_col=self.predicted_col,
        )
        val_dataset = SequenceClassifierDataset(
            df=validation_data,
            sequence_id_col=sequence_id_col,
            feature_cols=self.feature_cols,
            label_col=self.predicted_col,
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )
        early_stopping = EarlyStopping(
            patience=15, min_delta=0.001, restore_best_weights=True
        )

        logger.info("Start training...")
        for epoch in range(n_epochs):
            self.model.train()
            epoch_total_loss = 0
            epoch_correct_predictions = 0
            epoch_total_examples = 0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = (
                    batch_data.to(device),
                    batch_labels.to(device),
                )

                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = loss_fn(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                scheduler.step()

                epoch_total_loss += loss.item()
                epoch_total_examples += batch_labels.size(0)
                epoch_correct_predictions += self.get_n_correct_preds(
                    outputs=outputs, labels=batch_labels
                )
            epoch_avg_loss = epoch_total_loss / len(train_loader)
            epoch_train_accuracy = (
                100 * epoch_correct_predictions / epoch_total_examples
            )

            self.model.eval()
            epoch_val_total_loss = 0
            epoch_val_correct_predictions = 0
            epoch_val_total_examples = 0
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data, batch_labels = (
                        batch_data.to(device),
                        batch_labels.to(device),
                    )
                    outputs = self.model(batch_data)

                    epoch_val_total_loss += loss_fn(outputs, batch_labels).item()
                    epoch_val_total_examples += batch_labels.size(0)
                    epoch_val_correct_predictions += self.get_n_correct_preds(
                        outputs=outputs, labels=batch_labels
                    )
            epoch_avg_val_loss = epoch_val_total_loss / len(val_loader)
            epoch_val_accuracy = (
                100 * epoch_val_correct_predictions / epoch_val_total_examples
            )

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch: {epoch + 1}, "
                    f"Train Loss: {epoch_avg_loss:.4f}, "
                    f"Train Accuracy: {epoch_train_accuracy:.2f}%. "
                    f"Val Loss: {epoch_avg_val_loss:.4f}, "
                    f"Val Accuracy: {epoch_val_accuracy:.2f}%"
                )
            if use_early_stopping and early_stopping(epoch_avg_val_loss, self.model):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.2f}")
                break

    @staticmethod
    def load_model(save_dir_path: Path, file_name: str):
        with open(save_dir_path / f"{file_name}{MODEL_PATH_SUFFIX}", "rb") as f:
            model = pickle.load(f)
        model.model.load_state_dict(
            torch.load(
                save_dir_path / f"{file_name}{MODEL_WEIGHTS_PATH_SUFFIX}",
                weights_only=True,
            )
        )
        return model

    @staticmethod
    def get_n_correct_preds(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == labels).sum().item()

    def save_model(self, save_dir_path: Path, file_name: str):
        logger.info(f"Saving model to directory {save_dir_path}")
        save_dir_path.mkdir(exist_ok=True)
        torch.save(
            self.model.state_dict(),
            save_dir_path / f"{file_name}{MODEL_WEIGHTS_PATH_SUFFIX}",
        )
        with open(save_dir_path / f"{file_name}{MODEL_PATH_SUFFIX}", "wb") as f:
            pickle.dump(self, f)
        with open(save_dir_path / f"{file_name}_label_encoder.pkl", "wb") as f:
            pickle.dump(self.model.label_encoder, f)
        with open(save_dir_path / f"{file_name}_scaler.pkl", "wb") as f:
            pickle.dump(self.model.scaler, f)

    def scale_features(
        self, df: pl.DataFrame, fit_scaler: bool = False
    ) -> pl.DataFrame:
        if fit_scaler:
            scaled_features = self.model.scaler.fit_transform(
                df.select(self.feature_cols).to_numpy()
            )
        else:
            scaled_features = self.model.scaler.transform(
                df.select(self.feature_cols).to_numpy()
            )

        df = df.with_columns(
            [
                pl.Series(col, scaled_features[:, i])
                for i, col in enumerate(self.feature_cols)
            ]
        )
        return df

    def predict(
        self, sequence_data: pl.DataFrame, prediction_as_int: bool = True
    ) -> Union[int, str]:
        sequence_data = self.scale_features(df=sequence_data, fit_scaler=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            features = (
                torch.FloatTensor(
                    sequence_data.select(self.feature_cols).to_numpy().T.copy()
                )
                .unsqueeze(0)
                .to(device)
            )
            outputs = self.model(features)
            _, predicted = torch.max(outputs.data, 1)

        if prediction_as_int:
            prediction = int(predicted.cpu())
        else:
            prediction = self.model.label_encoder.inverse_transform(
                predicted.cpu()
            ).tolist()[0]
        return prediction

    def evaluate_accuracy(self, df: pl.DataFrame, sequence_id_col: str) -> float:
        df = df.with_columns(
            pl.Series(
                self.predicted_col,
                self.model.label_encoder.transform(df[self.predicted_col]),
            )
        )
        correct_predictions = 0
        total_examples = 0
        for seq_id, seq_data in df.group_by(sequence_id_col):
            prediction = self.predict(seq_data)
            actual_label = seq_data[self.predicted_col][0]
            if prediction == actual_label:
                correct_predictions += 1
            total_examples += 1
        accuracy = 100 * correct_predictions / total_examples
        return accuracy
