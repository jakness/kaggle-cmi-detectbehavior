import unittest

import polars as pl

from kaggle_cmi.data.utils import get_data_without_missing_values


class TestGetDataWithoutMissingValues(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence_id_col = "sequence_id"
        self.predicted_col = "label"
        self.f1 = "feat1"
        self.f2 = "feat2"
        self.f3 = "feat3"
        self.feature_cols = [self.f1, self.f2, self.f3]

    def test_drops_rows_where_either_key_column_is_null(self):
        df = pl.DataFrame(
            {
                self.sequence_id_col: ["A", "B", None],
                # predicted column uses strings
                self.predicted_col: ["0", None, None],
                self.f1: [1.0, 2.0, 3.0],
                self.f2: [10.0, 20.0, 30.0],
                self.f3: [100.0, 200.0, 300.0],
            }
        )

        result = get_data_without_missing_values(
            df=df,
            feature_cols_to_check=self.feature_cols,
            predicted_col_to_check=self.predicted_col,
            sequence_id_col_to_check=self.sequence_id_col,
            accept_rows_with_pct_feature_cols_missing_values=0.0,
            missing_features_fill_value=-1,
        )

        # With any_horizontal: drop rows where *either* key col is null
        # Row B dropped (predicted_col null), Row C dropped (both null)
        self.assertEqual(len(result), 1)
        self.assertListEqual(result[self.sequence_id_col].to_list(), ["A"])
        self.assertListEqual(result[self.predicted_col].to_list(), ["0"])

    def test_filters_rows_by_missing_feature_threshold_and_fills_with_stricter_key_check(
        self,
    ):
        df = pl.DataFrame(
            {
                self.sequence_id_col: ["A", "B", "C", None],
                # predicted column uses strings
                self.predicted_col: ["1", None, "3", None],
                # Row-wise feature null counts: A=0, B=1, C=2, D=0
                self.f1: [1.0, None, None, 4.0],
                self.f2: [2.0, 5.0, None, 8.0],
                self.f3: [3.0, 6.0, 9.0, 12.0],
            }
        )

        fill_value = -999
        result = get_data_without_missing_values(
            df=df,
            feature_cols_to_check=self.feature_cols,
            predicted_col_to_check=self.predicted_col,
            sequence_id_col_to_check=self.sequence_id_col,
            accept_rows_with_pct_feature_cols_missing_values=0.2,
            missing_features_fill_value=fill_value,
        )

        self.assertEqual(len(result), 1)
        self.assertListEqual(result[self.sequence_id_col].to_list(), ["A"])
        self.assertListEqual(result[self.predicted_col].to_list(), ["1"])

        # No nulls in feature cols for surviving row
        for col in self.feature_cols:
            self.assertEqual(result[col].null_count(), 0)


if __name__ == "__main__":
    unittest.main()
