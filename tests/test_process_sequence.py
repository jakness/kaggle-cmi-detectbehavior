import unittest

import polars as pl

from kaggle_cmi.data.utils import process_sequence


class TestProcessSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.feat_1_str = "feat1"
        self.feat_2_str = "feat2"
        self.sequence_id_str = "sequence_id"
        self.label_str = "label"
        self.feature_cols = [self.feat_1_str, self.feat_2_str]
        self.non_feature_cols = [self.sequence_id_str, self.label_str]

    def build_sequence_df(self, num_rows: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                self.sequence_id_str: ["A"] * num_rows,
                self.label_str: [1] * num_rows,
                self.feat_1_str: [float(i) for i in range(num_rows)],
                self.feat_2_str: [float(i * 10) for i in range(num_rows)],
            }
        )

    def test_truncate_when_longer_than_target_length(self):
        original = self.build_sequence_df(num_rows=5)
        seq_len = 3
        result = process_sequence(
            df=original,
            feature_cols=self.feature_cols,
            sequence_length=seq_len,
            features_pad_value=0.0,
        )

        # Expect the last 3 rows preserved
        self.assertEqual(len(result), seq_len)
        self.assertListEqual(
            result[self.feat_1_str].to_list(),
            original[self.feat_1_str][-seq_len:].to_list(),
        )
        self.assertListEqual(
            result[self.feat_2_str].to_list(),
            original[self.feat_2_str][-seq_len:].to_list(),
        )
        self.assertListEqual(
            result[self.sequence_id_str].to_list(),
            original[self.sequence_id_str][-seq_len:].to_list(),
        )
        self.assertListEqual(
            result[self.label_str].to_list(),
            original[self.label_str][-seq_len:].to_list(),
        )

    def test_pad_when_shorter_than_target_length(self):
        original = self.build_sequence_df(num_rows=2)
        pad_value = -1.0
        target_len = 5
        result = process_sequence(
            df=original,
            feature_cols=self.feature_cols,
            sequence_length=target_len,
            features_pad_value=pad_value,
        )

        self.assertEqual(len(result), target_len)

        pad_len = target_len - len(original)

        self.assertListEqual(
            result[self.feat_1_str][:pad_len].to_list(), [pad_value] * pad_len
        )
        self.assertListEqual(
            result[self.feat_2_str][:pad_len].to_list(), [pad_value] * pad_len
        )

        self.assertListEqual(
            result[self.sequence_id_str][:pad_len].to_list(),
            [original[self.sequence_id_str][0]] * pad_len,
        )
        self.assertListEqual(
            result[self.label_str][:pad_len].to_list(),
            [original[self.label_str][0]] * pad_len,
        )

        self.assertListEqual(
            result[self.feat_1_str][pad_len:].to_list(),
            original[self.feat_1_str].to_list(),
        )
        self.assertListEqual(
            result[self.feat_2_str][pad_len:].to_list(),
            original[self.feat_2_str].to_list(),
        )
        self.assertListEqual(
            result[self.sequence_id_str][pad_len:].to_list(),
            original[self.sequence_id_str].to_list(),
        )
        self.assertListEqual(
            result[self.label_str][pad_len:].to_list(),
            original[self.label_str].to_list(),
        )

    def test_no_change_when_equal_length(self):
        len_seq = 4
        original = self.build_sequence_df(num_rows=len_seq)
        result = process_sequence(
            df=original,
            feature_cols=self.feature_cols,
            sequence_length=len_seq,
            features_pad_value=0.0,
        )

        self.assertEqual(len(result), len_seq)
        for col in self.non_feature_cols + self.feature_cols:
            self.assertListEqual(result[col].to_list(), original[col].to_list())


if __name__ == "__main__":
    unittest.main()
