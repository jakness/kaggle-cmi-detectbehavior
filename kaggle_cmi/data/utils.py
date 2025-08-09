import logging
from typing import List, Tuple

import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class SequenceClassifierDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        sequence_id_col: str,
        feature_cols: List[str],
        label_col: str,
    ):
        self.sequences = []
        self.labels = []

        for seq_id, sequence in df.group_by(sequence_id_col):
            self.sequences.append(sequence[feature_cols].to_numpy().T)
            self.labels.append(sequence[label_col][0])

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_data_without_missing_values(
    df: pl.DataFrame,
    feature_cols_to_check: List[str],
    predicted_col_to_check: str,
    sequence_id_col_to_check: str,
    accept_rows_with_pct_feature_cols_missing_values: float,
    missing_features_fill_value: float,
) -> pl.DataFrame:
    df_filtered = df.filter(
        ~pl.any_horizontal(
            pl.col([predicted_col_to_check, sequence_id_col_to_check]).is_null()
        )
    )
    df_filtered = df_filtered.filter(
        pl.sum_horizontal(pl.col(feature_cols_to_check).is_null())
        <= int(len(df.columns) * accept_rows_with_pct_feature_cols_missing_values)
    )
    df_filtered = df_filtered.fill_null(missing_features_fill_value)
    return df_filtered


def process_sequence(
    df: pl.DataFrame,
    feature_cols: List[str],
    sequence_length: int,
    features_pad_value: float,
) -> pl.DataFrame:
    sequence_data = df
    original_seq_len = len(sequence_data)
    if original_seq_len > sequence_length:
        sequence_data = sequence_data[-sequence_length:]
    elif original_seq_len < sequence_length:
        pad_length = sequence_length - original_seq_len
        pad_features = {col: [features_pad_value] * pad_length for col in feature_cols}
        pad_other_cols = {
            col: [sequence_data[col][0]] * pad_length
            for col in sequence_data.columns
            if col not in feature_cols
        }
        pad_sequence_data = pl.DataFrame({**pad_other_cols, **pad_features})
        sequence_data = pad_sequence_data.vstack(sequence_data)
    return sequence_data


def get_processed_sequences(
    df: pl.DataFrame,
    feature_cols: List[str],
    sequence_id_col: str,
    sequence_length: int,
) -> pl.DataFrame:
    processed_sequences = []
    for sequence_id, sequence_data in df.group_by(sequence_id_col):
        processed_sequences.append(
            process_sequence(
                df=sequence_data,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
                features_pad_value=0.0,
            )
        )
    return pl.concat(processed_sequences)


def get_preprocessed_training_data(
    df: pl.DataFrame,
    feature_cols: List[str],
    predicted_col: str,
    sequence_id_col: str,
    sequence_length: int,
    fill_val_for_missing_and_nan: float,
) -> pl.DataFrame:
    logger.info("Processing training data...")
    df = df.select([sequence_id_col, predicted_col] + feature_cols)
    df = get_data_without_missing_values(
        df=df,
        feature_cols_to_check=feature_cols,
        predicted_col_to_check=predicted_col,
        sequence_id_col_to_check=sequence_id_col,
        accept_rows_with_pct_feature_cols_missing_values=0.05,
        missing_features_fill_value=fill_val_for_missing_and_nan,
    )
    df = get_processed_sequences(
        df=df,
        feature_cols=feature_cols,
        sequence_id_col=sequence_id_col,
        sequence_length=sequence_length,
    )
    return df


def get_train_valid_test_split(
    df: pl.DataFrame,
    sequence_id_col: str,
    predicted_col: str,
    val_size: float,
    test_size: float,
    use_only_valid: bool,
    random_state: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    logger.info("Splitting data into train/validation/test sets...")
    sequence_ids_with_predicted_class = (
        df.select([sequence_id_col, predicted_col])
        .unique(subset=[sequence_id_col], keep="first")
        .to_pandas()
    )

    train_sequence_ids, val_sequence_ids = train_test_split(
        sequence_ids_with_predicted_class[sequence_id_col],
        stratify=sequence_ids_with_predicted_class[predicted_col],
        test_size=val_size if use_only_valid else val_size + test_size,
        random_state=random_state,
    )

    if use_only_valid:
        test_sequence_ids = []
    else:
        val_test_sequence_ids_with_predicted_class = sequence_ids_with_predicted_class[
            sequence_ids_with_predicted_class[sequence_id_col].isin(val_sequence_ids)
        ]
        val_sequence_ids, test_sequence_ids = train_test_split(
            val_test_sequence_ids_with_predicted_class[sequence_id_col],
            stratify=val_test_sequence_ids_with_predicted_class[predicted_col],
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
        )

    train_df = df.filter(pl.col(sequence_id_col).is_in(train_sequence_ids))
    val_df = df.filter(pl.col(sequence_id_col).is_in(val_sequence_ids))
    test_df = df.filter(pl.col(sequence_id_col).is_in(test_sequence_ids))
    return train_df, val_df, test_df


def get_feature_cols_with_prefix(df: pl.DataFrame, prefixes: List[str]) -> List[str]:
    return [col for prefix in prefixes for col in df.columns if col.startswith(prefix)]
