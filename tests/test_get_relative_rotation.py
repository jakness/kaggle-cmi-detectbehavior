import unittest
import math

import numpy as np

from kaggle_cmi.data.constants import (
    COL_ROT_W,
    COL_ROT_X,
    COL_ROT_Y,
    COL_ROT_Z,
    COL_PREV_SUFFIX,
)
from kaggle_cmi.models.features import get_relative_rotation


class TestGetRelativeRotation(unittest.TestCase):
    def test_identity_to_identity(self):
        row = {
            COL_ROT_W + COL_PREV_SUFFIX: 1.0,
            COL_ROT_X + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Y + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Z + COL_PREV_SUFFIX: 0.0,
            COL_ROT_W: 1.0,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        r_rel = get_relative_rotation(row)
        np.testing.assert_allclose(
            r_rel.as_quat(scalar_first=True), [1, 0, 0, 0], rtol=1e-6, atol=1e-12
        )

    def test_rotation_to_same_rotation(self):
        sqrt_half = math.sqrt(0.5)
        row = {
            COL_ROT_W + COL_PREV_SUFFIX: sqrt_half,
            COL_ROT_X + COL_PREV_SUFFIX: sqrt_half,
            COL_ROT_Y + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Z + COL_PREV_SUFFIX: 0.0,
            COL_ROT_W: sqrt_half,
            COL_ROT_X: sqrt_half,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        r_rel = get_relative_rotation(row)
        np.testing.assert_allclose(
            r_rel.as_quat(scalar_first=True), [1, 0, 0, 0], rtol=1e-6, atol=1e-12
        )

    def test_identity_to_90deg_x(self):
        sqrt_half = math.sqrt(0.5)
        row = {
            COL_ROT_W + COL_PREV_SUFFIX: 1.0,
            COL_ROT_X + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Y + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Z + COL_PREV_SUFFIX: 0.0,
            COL_ROT_W: sqrt_half,
            COL_ROT_X: sqrt_half,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        r_rel = get_relative_rotation(row)
        axis, angle = (
            r_rel.as_rotvec() / np.linalg.norm(r_rel.as_rotvec()),
            np.linalg.norm(r_rel.as_rotvec()),
        )
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(angle, math.pi / 2, places=6)

    def test_invalid_value_returns_none(self):
        row = {
            COL_ROT_W + COL_PREV_SUFFIX: float("nan"),
            COL_ROT_X + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Y + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Z + COL_PREV_SUFFIX: 0.0,
            COL_ROT_W: 1.0,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        self.assertIsNone(get_relative_rotation(row))

    def test_zero_norm_returns_none(self):
        row = {
            COL_ROT_W + COL_PREV_SUFFIX: 0.0,
            COL_ROT_X + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Y + COL_PREV_SUFFIX: 0.0,
            COL_ROT_Z + COL_PREV_SUFFIX: 0.0,
            COL_ROT_W: 1.0,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        self.assertIsNone(get_relative_rotation(row))


if __name__ == "__main__":
    unittest.main()
