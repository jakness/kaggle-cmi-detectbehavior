import logging

import polars as pl

from kaggle_cmi.data.constants import (
    DATA_RAW_DIR_PATH,
    COL_SEQUENCE_ID,
    COL_GESTURE,
    COL_PHASE,
    COL_SUBJECT,
)


logger = logging.getLogger(__name__)


def analyze_dataframe(df: pl.DataFrame):
    logger.info(f"Shape of the data: {df.shape}")
    logger.info(f"Columns in the data: {df.columns}")
    logger.info(f"Schema of the data: {df.schema}\n")

    is_missing = any(df.select(pl.all().is_null().any()).row(0))
    if is_missing:
        logger.info(
            "There are missing values in the data. Number of missing values in the columns:"
        )
        n_missing = df.select(
            [
                pl.col(col).is_null().sum().alias(f"{col}_n_missing")
                for col in df.columns
            ]
        )
        logging.info(f"{n_missing.to_dict(as_series=False)}\n")

    duplicate_count = len(df) - len(df.unique())
    logger.info(f"Duplicate rows: {duplicate_count}\n")

    logger.info(
        f"Data is collected from {len(df[COL_SUBJECT].unique())} different subjects.\n"
    )

    subjects = df[COL_SUBJECT].unique()
    for subject in subjects:
        subject_df = df.filter(pl.col(COL_SUBJECT) == subject)
        logger.info(
            f"Number of collected sequences by subject '{subject}': {len(subject_df[COL_SEQUENCE_ID].unique())}"
        )

    gestures = df[COL_GESTURE].unique()
    for gesture in gestures:
        gesture_df = df.filter(pl.col(COL_GESTURE) == gesture)
        logger.info(
            f"Number of collected sequences for gesture '{gesture}' in the data: {len(gesture_df[COL_SEQUENCE_ID].unique())}"
        )


def analyze_sequences(df: pl.DataFrame):
    seq_stats = df.group_by(COL_SEQUENCE_ID).agg([pl.len().alias("sequence_length")])
    logger.info(f"Total number of sequences: {len(seq_stats)}")
    logger.info("Sequence length statistics:")
    logger.info(seq_stats.describe())

    sequence_phases = df[COL_PHASE].unique()
    gestures = df[COL_GESTURE].unique()
    for phase in sequence_phases:
        logger.info(f"***** Info on sequence phase '{phase}' *****")
        logger.info("Phase statistics:")
        df_phase = df.filter(pl.col(COL_PHASE) == phase)
        phase_stats = df_phase.group_by(COL_SEQUENCE_ID).agg(
            [
                pl.len().alias(f"{phase}_phase_length"),
            ]
        )
        logger.info(phase_stats.describe())
        for gesture in gestures:
            logger.info(f"** Info on gesture '{gesture}' in the phase **")
            df_gesture = df_phase.filter(pl.col(COL_GESTURE) == gesture)
            gesture_stats = df_gesture.group_by(COL_SEQUENCE_ID).agg(
                [pl.len().alias(f"{gesture}_gesture_length")]
            )
            logger.info("Gesture statistics:")
            logger.info(gesture_stats.describe())


def analyze_sensor_correlations(df: pl.DataFrame):
    numeric_df = df.select(
        [
            col
            for col in df.columns
            if df[col].dtype.is_numeric() and col != "sequence_counter"
        ]
    )
    numeric_df = numeric_df.drop_nulls()
    correlation_matrix = numeric_df.corr().to_pandas()
    correlation_matrix.index = correlation_matrix.columns
    corr_threshold = 0.85
    high_correlation = (
        correlation_matrix.stack()[correlation_matrix.stack() > corr_threshold]
        .reset_index()
        .rename(columns={"level_0": "row", "level_1": "column", 0: "value"})
    )
    filtered = high_correlation[high_correlation["row"] != high_correlation["column"]]
    logger.info(f"Correlations over {corr_threshold}:")
    logger.info(f"{filtered}")


def main():
    df_train = pl.read_csv(DATA_RAW_DIR_PATH / "train.csv")
    logger.info("Analyzing training data")
    analyze_dataframe(df_train)
    analyze_sequences(df_train)
    analyze_sensor_correlations(df_train)


if __name__ == "__main__":
    main()
