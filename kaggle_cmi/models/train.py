import logging
import argparse
from pathlib import Path

import polars as pl

from kaggle_cmi.data.constants import (
    DATA_RAW_DIR_PATH,
    COL_PREFIX_ACC,
    COL_PREFIX_ROT,
    COL_PREFIX_TEMP,
    COL_PREFIX_TOF,
    COL_SEQUENCE_ID,
    COL_GESTURE,
)
from kaggle_cmi.data.utils import (
    get_preprocessed_training_data,
    get_train_valid_test_split,
    get_feature_cols_with_prefix,
)
from kaggle_cmi.models.features import get_feature_engineered_data
from kaggle_cmi.models.conv1d_model import Conv1DSequenceClassifier
from kaggle_cmi.models.two_branch_model import TwoBranchSequenceClassifier
from kaggle_cmi.models.se_gru import SqueezeExcitationGRUSequenceClassifier
from kaggle_cmi.models.utils import get_model


logger = logging.getLogger(__name__)


def main(
    model: str,
    use_only_imu: bool,
    use_only_valid: bool,
    use_early_stopping: bool,
    n_epochs: int,
    sequence_length: int,
    saved_model_dir_path: Path,
    saved_model_name: str,
):
    sequence_classifier = get_model(name=model)
    train_data_raw = pl.read_csv(DATA_RAW_DIR_PATH / "train.csv")
    fill_value = 0.0

    imu_feature_cols = get_feature_cols_with_prefix(
        df=train_data_raw, prefixes=[COL_PREFIX_ACC, COL_PREFIX_ROT]
    )
    thm_feature_cols = []
    tof_feature_cols = []
    if not use_only_imu:
        thm_feature_cols = get_feature_cols_with_prefix(
            df=train_data_raw, prefixes=[COL_PREFIX_TEMP]
        )
        tof_feature_cols = get_feature_cols_with_prefix(
            df=train_data_raw, prefixes=[COL_PREFIX_TOF]
        )
    feature_cols = imu_feature_cols + thm_feature_cols + tof_feature_cols

    train_data = get_preprocessed_training_data(
        df=train_data_raw,
        feature_cols=feature_cols,
        predicted_col=COL_GESTURE,
        sequence_id_col=COL_SEQUENCE_ID,
        sequence_length=sequence_length,
        fill_val_for_missing_and_nan=fill_value,
    )

    train_data, imu_feature_cols, thm_feature_cols, tof_feature_cols, feature_cols = (
        get_feature_engineered_data(
            df=train_data,
            imu_feature_cols=imu_feature_cols,
            thm_feature_cols=thm_feature_cols,
            tof_feature_cols=tof_feature_cols,
            feature_cols=feature_cols,
            sequence_id_col=COL_SEQUENCE_ID,
            fill_val_for_missing_and_nan=fill_value,
        )
    )

    train_data, valid_data, test_data = get_train_valid_test_split(
        df=train_data,
        sequence_id_col=COL_SEQUENCE_ID,
        predicted_col=COL_GESTURE,
        val_size=0.15,
        test_size=0.15,
        use_only_valid=use_only_valid,
    )

    sequence_classifier = sequence_classifier(
        feature_cols=feature_cols,
        imu_feature_cols=imu_feature_cols,
        thm_tof_feature_cols=thm_feature_cols + tof_feature_cols,
        predicted_col=COL_GESTURE,
        n_classes=len(train_data[COL_GESTURE].unique()),
        sequence_length=sequence_length,
    )
    sequence_classifier.train_model(
        train_data=train_data,
        validation_data=valid_data,
        sequence_id_col=COL_SEQUENCE_ID,
        n_epochs=n_epochs,
        use_early_stopping=use_early_stopping,
    )
    sequence_classifier.save_model(
        save_dir_path=saved_model_dir_path, file_name=saved_model_name
    )

    if not use_only_valid:
        test_accuracy = sequence_classifier.evaluate_accuracy(
            df=test_data, sequence_id_col=COL_SEQUENCE_ID
        )
        logger.info(f"Test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequence classifier model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to train",
        choices=[
            Conv1DSequenceClassifier.name,
            TwoBranchSequenceClassifier.name,
            SqueezeExcitationGRUSequenceClassifier.name,
        ],
    )
    parser.add_argument(
        "--use_only_imu",
        action="store_true",
        help="Use only IMU features (accelerometer and rotation)",
    )
    parser.add_argument(
        "--use_only_valid",
        action="store_true",
        help="Use only validation split (no test split)",
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Use early stopping during training",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=80,
        help="Length of a input sequence for the model",
    )
    parser.add_argument(
        "--saved_model_dir_path",
        type=Path,
        required=True,
        help="Directory path to save the trained model",
    )
    parser.add_argument(
        "--saved_model_name", type=str, required=True, help="Name for the saved model"
    )

    args = parser.parse_args()

    main(**vars(args))
