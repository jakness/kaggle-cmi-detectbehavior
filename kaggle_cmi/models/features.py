import math
import logging
from typing import List, Tuple, Union, Dict

import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation

from kaggle_cmi.data.constants import (
    COL_PREV_SUFFIX,
    COL_ACC_X,
    COL_ACC_Y,
    COL_ACC_Z,
    COL_ACC_MAGNITUDE,
    COL_ACC_MAGNITUDE_JERK,
    COL_LINEAR_ACC,
    COL_LINEAR_ACC_X,
    COL_LINEAR_ACC_Y,
    COL_LINEAR_ACC_Z,
    COL_LINEAR_ACC_MAGNITUDE,
    COL_LINEAR_ACC_MAGNITUDE_JERK,
    COL_ROT_W,
    COL_ROT_X,
    COL_ROT_Y,
    COL_ROT_Z,
    COL_ROT_ANGLE,
    COL_ROT_ANGLE_VELOCITY,
    COL_ANGULAR_VELOCITY,
    COL_ANGULAR_VELOCITY_X,
    COL_ANGULAR_VELOCITY_Y,
    COL_ANGULAR_VELOCITY_Z,
    COL_ANGULAR_VELOCITY_MAGNITUDE,
    COL_ANGULAR_VELOCITY_MAGNITUDE_JERK,
    COL_ANGULAR_VELOCITY_X_JERK,
    COL_ANGULAR_VELOCITY_Y_JERK,
    COL_ANGULAR_VELOCITY_Z_JERK,
    COL_ANGULAR_VELOCITY_MAGNITUDE_SNAP,
    COL_ANGULAR_VELOCITY_X_SNAP,
    COL_ANGULAR_VELOCITY_Y_SNAP,
    COL_ANGULAR_VELOCITY_Z_SNAP,
    COL_ANGULAR_DISTANCE,
    N_TOF_SENSORS,
    COL_TOF_MEAN,
    COL_TOF_MAX,
    COL_TOF_MIN,
)


logger = logging.getLogger(__name__)


def is_invalid(value: float):
    return value is None or math.isnan(value)


def get_linear_acceleration(row: Dict[str, float]) -> Union[List[float], List[None]]:
    acc = np.array([row[COL_ACC_X], row[COL_ACC_Y], row[COL_ACC_Z]])
    q = np.array([row[COL_ROT_W], row[COL_ROT_X], row[COL_ROT_Y], row[COL_ROT_Z]])

    if any(is_invalid(val) for val in q):
        return [None, None, None]

    norm = np.linalg.norm(q)
    if norm == 0 or math.isnan(norm):
        return [None, None, None]
    q /= norm

    r = Rotation.from_quat(quat=q, scalar_first=True)

    gravity_world = np.array([0, 0, 9.81])

    gravity_sensor_frame = r.apply(gravity_world, inverse=True)

    lin_acc = acc - gravity_sensor_frame

    return lin_acc.tolist()


def get_relative_rotation(row: Dict[str, float]) -> Union[np.array, None]:
    q_prev = np.array(
        [
            row[COL_ROT_W + COL_PREV_SUFFIX],
            row[COL_ROT_X + COL_PREV_SUFFIX],
            row[COL_ROT_Y + COL_PREV_SUFFIX],
            row[COL_ROT_Z + COL_PREV_SUFFIX],
        ]
    )
    q_current = np.array(
        [row[COL_ROT_W], row[COL_ROT_X], row[COL_ROT_Y], row[COL_ROT_Z]]
    )

    if any(is_invalid(val) for val in np.append(q_prev, q_current)):
        return None

    norm_prev = np.linalg.norm(q_prev)
    norm_current = np.linalg.norm(q_current)
    if (
        norm_prev == 0
        or norm_current == 0
        or math.isnan(norm_prev)
        or math.isnan(norm_current)
    ):
        return None
    q_prev /= norm_prev
    q_current /= norm_current

    r_prev = Rotation.from_quat(quat=q_prev, scalar_first=True)
    r_current = Rotation.from_quat(quat=q_current, scalar_first=True)

    r_rel = r_current * r_prev.inv()
    return r_rel


def get_angular_velocity(
    row: Dict[str, float], dt: float = 0.05
) -> Union[List[float], List[None]]:
    r_rel = get_relative_rotation(row=row)
    if r_rel is None:
        return [None, None, None]
    axis_angle = r_rel.as_rotvec()

    angular_velocity = axis_angle / dt
    return angular_velocity.tolist()


def get_angular_distance(row: Dict[str, float]) -> Union[float, None]:
    r_rel = get_relative_rotation(row=row)
    if r_rel is None:
        return None
    angular_distance = np.linalg.norm(r_rel.as_rotvec())

    return float(angular_distance)


def get_diff_feature(
    df: pl.DataFrame, col: str, sequence_id_col: str, feat_name: str
) -> pl.DataFrame:
    df = df.with_columns([pl.col(col).diff().over(sequence_id_col).alias(feat_name)])
    return df


def create_shifted_columns(
    df: pl.DataFrame, cols: List[str], sequence_id_col: str
) -> Tuple[pl.DataFrame, List[str]]:
    shifted_cols = []
    for col in cols:
        shifted_col = f"{col}{COL_PREV_SUFFIX}"
        shifted_cols.append(shifted_col)
        df = df.with_columns(
            [pl.col(col).shift(1).over(sequence_id_col).alias(shifted_col)]
        )
    return df, shifted_cols


def calculate_acceleration_features(
    df: pl.DataFrame, sequence_id_col: str
) -> pl.DataFrame:
    df = df.with_columns(
        [
            (
                (
                    pl.col(COL_ACC_X) ** 2
                    + pl.col(COL_ACC_Y) ** 2
                    + pl.col(COL_ACC_Z) ** 2
                ).sqrt()
            ).alias(COL_ACC_MAGNITUDE)
        ]
    )
    df = get_diff_feature(
        df=df,
        col=COL_ACC_MAGNITUDE,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ACC_MAGNITUDE_JERK,
    )
    return df


def calculate_rotation_angle_features(
    df: pl.DataFrame, sequence_id_col: str
) -> pl.DataFrame:
    df = df.with_columns(
        [
            (
                2
                * (
                    pl.col(COL_ROT_W)
                    / (
                        (
                            pl.col(COL_ROT_W) ** 2
                            + pl.col(COL_ROT_X) ** 2
                            + pl.col(COL_ROT_Y) ** 2
                            + pl.col(COL_ROT_Z) ** 2
                        ).sqrt()
                    )
                ).arccos()
            ).alias(COL_ROT_ANGLE)
        ]
    )
    df = get_diff_feature(
        df=df,
        col=COL_ROT_ANGLE,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ROT_ANGLE_VELOCITY,
    )
    return df


def calculate_linear_acceleration_features(
    df: pl.DataFrame, sequence_id_col: str
) -> pl.DataFrame:
    # linear acceleration (gravity removed from accelerometer data)
    df = df.with_columns(
        [
            pl.struct(
                [
                    COL_ACC_X,
                    COL_ACC_Y,
                    COL_ACC_Z,
                    COL_ROT_W,
                    COL_ROT_X,
                    COL_ROT_Y,
                    COL_ROT_Z,
                ]
            )
            .map_elements(get_linear_acceleration, return_dtype=pl.List(pl.Float64))
            .alias(COL_LINEAR_ACC)
        ]
    )

    df = df.with_columns(
        [
            pl.col(COL_LINEAR_ACC).list.get(0).alias(COL_LINEAR_ACC_X),
            pl.col(COL_LINEAR_ACC).list.get(1).alias(COL_LINEAR_ACC_Y),
            pl.col(COL_LINEAR_ACC).list.get(2).alias(COL_LINEAR_ACC_Z),
        ]
    ).drop(COL_LINEAR_ACC)
    df = df.with_columns(
        [
            (
                (
                    pl.col(COL_LINEAR_ACC_X) ** 2
                    + pl.col(COL_LINEAR_ACC_Y) ** 2
                    + pl.col(COL_LINEAR_ACC_Z) ** 2
                ).sqrt()
            ).alias(COL_LINEAR_ACC_MAGNITUDE)
        ]
    )
    df = get_diff_feature(
        df=df,
        col=COL_LINEAR_ACC_MAGNITUDE,
        sequence_id_col=sequence_id_col,
        feat_name=COL_LINEAR_ACC_MAGNITUDE_JERK,
    )
    return df


def calculate_angular_velocity_features(
    df: pl.DataFrame, sequence_id_col: str
) -> pl.DataFrame:
    df, shifted_cols = create_shifted_columns(
        df=df,
        cols=[COL_ROT_W, COL_ROT_X, COL_ROT_Y, COL_ROT_Z],
        sequence_id_col=sequence_id_col,
    )
    df = df.with_columns(
        [
            pl.struct([COL_ROT_W, COL_ROT_X, COL_ROT_Y, COL_ROT_Z] + shifted_cols)
            .map_elements(get_angular_velocity, return_dtype=pl.List(pl.Float64))
            .alias(COL_ANGULAR_VELOCITY)
        ]
    )
    df = df.with_columns(
        [
            pl.col(COL_ANGULAR_VELOCITY).list.get(0).alias(COL_ANGULAR_VELOCITY_X),
            pl.col(COL_ANGULAR_VELOCITY).list.get(1).alias(COL_ANGULAR_VELOCITY_Y),
            pl.col(COL_ANGULAR_VELOCITY).list.get(2).alias(COL_ANGULAR_VELOCITY_Z),
        ]
    ).drop(COL_ANGULAR_VELOCITY)
    df = df.drop(shifted_cols)
    df = df.with_columns(
        [
            (
                (
                    pl.col(COL_ANGULAR_VELOCITY_X) ** 2
                    + pl.col(COL_ANGULAR_VELOCITY_Y) ** 2
                    + pl.col(COL_ANGULAR_VELOCITY_Z) ** 2
                ).sqrt()
            ).alias(COL_ANGULAR_VELOCITY_MAGNITUDE)
        ]
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_MAGNITUDE,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_MAGNITUDE_JERK,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_MAGNITUDE_JERK,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_MAGNITUDE_SNAP,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_X,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_X_JERK,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_Y,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_Y_JERK,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_Z,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_Z_JERK,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_X_JERK,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_X_SNAP,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_Y_JERK,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_Y_SNAP,
    )
    df = get_diff_feature(
        df=df,
        col=COL_ANGULAR_VELOCITY_Z_JERK,
        sequence_id_col=sequence_id_col,
        feat_name=COL_ANGULAR_VELOCITY_Z_SNAP,
    )
    return df


def calculate_angular_distance_feature(
    df: pl.DataFrame, sequence_id_col: str
) -> pl.DataFrame:
    df, shifted_cols = create_shifted_columns(
        df=df,
        cols=[COL_ROT_W, COL_ROT_X, COL_ROT_Y, COL_ROT_Z],
        sequence_id_col=sequence_id_col,
    )
    df = df.with_columns(
        [
            pl.struct([COL_ROT_W, COL_ROT_X, COL_ROT_Y, COL_ROT_Z] + shifted_cols)
            .map_elements(get_angular_distance, return_dtype=pl.Float64)
            .alias(COL_ANGULAR_DISTANCE)
        ]
    )
    df = df.drop(shifted_cols)
    return df


def calculate_tof_features(
    df: pl.DataFrame, tof_feature_cols: List[str], fill_value: float
) -> pl.DataFrame:
    df = df.with_columns(
        [pl.col(col).replace(-1, fill_value).alias(col) for col in tof_feature_cols]
    )
    for i in range(1, N_TOF_SENSORS + 1):
        tof_sensor_cols = [col for col in tof_feature_cols if f"_{i}_" in col]
        df = df.with_columns(
            [
                pl.mean_horizontal(pl.col(tof_sensor_cols)).alias(
                    f"{COL_TOF_MEAN}_{i}"
                ),
                pl.max_horizontal(pl.col(tof_sensor_cols)).alias(f"{COL_TOF_MAX}_{i}"),
                pl.min_horizontal(pl.col(tof_sensor_cols)).alias(f"{COL_TOF_MIN}_{i}"),
            ]
        )
    return df


def get_feature_engineered_data(
    df: pl.DataFrame,
    imu_feature_cols: List[str],
    thm_feature_cols: List[str],
    tof_feature_cols: List[str],
    feature_cols: List[str],
    sequence_id_col: str,
    fill_val_for_missing_and_nan: float,
):
    logger.info("Applying feature engineering...")
    non_feature_cols = [col for col in df.columns if col not in feature_cols]

    # imu features
    df = calculate_acceleration_features(df=df, sequence_id_col=sequence_id_col)
    df = calculate_rotation_angle_features(df=df, sequence_id_col=sequence_id_col)
    df = calculate_linear_acceleration_features(df=df, sequence_id_col=sequence_id_col)
    df = calculate_angular_velocity_features(df=df, sequence_id_col=sequence_id_col)
    df = calculate_angular_distance_feature(df=df, sequence_id_col=sequence_id_col)
    imu_feature_engineered_cols = [
        col for col in df.columns if col not in feature_cols + non_feature_cols
    ]
    imu_feature_cols = imu_feature_cols + imu_feature_engineered_cols

    # # tof features
    if tof_feature_cols:
        df = calculate_tof_features(
            df=df,
            tof_feature_cols=tof_feature_cols,
            fill_value=fill_val_for_missing_and_nan,
        )
        tof_feature_engineered_cols = [
            col
            for col in df.columns
            if col
            not in feature_cols + non_feature_cols + imu_feature_cols + thm_feature_cols
        ]
        tof_feature_cols = tof_feature_cols + tof_feature_engineered_cols

    df = df.fill_nan(fill_val_for_missing_and_nan).fill_null(
        fill_val_for_missing_and_nan
    )
    feature_cols = imu_feature_cols + thm_feature_cols + tof_feature_cols
    return (
        df.select(pl.col(non_feature_cols + feature_cols)),
        imu_feature_cols,
        thm_feature_cols,
        tof_feature_cols,
        feature_cols,
    )
