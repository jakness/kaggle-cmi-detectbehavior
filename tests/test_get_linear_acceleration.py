import unittest
import math

import numpy as np

from kaggle_cmi.data.constants import (
    COL_ACC_X,
    COL_ACC_Y,
    COL_ACC_Z,
    COL_ROT_W,
    COL_ROT_X,
    COL_ROT_Y,
    COL_ROT_Z,
)
from kaggle_cmi.models.features import get_linear_acceleration


class TestGetLinearAccelerationBlackBox(unittest.TestCase):
    def assert_zero_linear_acc(self, row):
        result = get_linear_acceleration(row)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], rtol=1e-6, atol=1e-12)

    def test_identity_quaternion(self):
        row = {
            COL_ACC_X: 0.0,
            COL_ACC_Y: 0.0,
            COL_ACC_Z: 9.81,
            COL_ROT_W: 1.0,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        self.assert_zero_linear_acc(row)

    def test_180deg_x_axis(self):
        row = {
            COL_ACC_X: 0.0,
            COL_ACC_Y: 0.0,
            COL_ACC_Z: -9.81,
            COL_ROT_W: 0.0,
            COL_ROT_X: 1.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        self.assert_zero_linear_acc(row)

    def test_90deg_y_axis(self):
        sqrt_half = math.sqrt(0.5)
        row = {
            COL_ACC_X: -9.81,
            COL_ACC_Y: 0.0,
            COL_ACC_Z: 0.0,
            COL_ROT_W: sqrt_half,
            COL_ROT_X: 0.0,
            COL_ROT_Y: sqrt_half,
            COL_ROT_Z: 0.0,
        }
        self.assert_zero_linear_acc(row)

    def test_90deg_x_axis(self):
        sqrt_half = math.sqrt(0.5)
        row = {
            COL_ACC_X: 0.0,
            COL_ACC_Y: 9.81,
            COL_ACC_Z: 0.0,
            COL_ROT_W: sqrt_half,
            COL_ROT_X: sqrt_half,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        self.assert_zero_linear_acc(row)

    def test_90deg_z_axis(self):
        sqrt_half = math.sqrt(0.5)
        row = {
            COL_ACC_X: 0.0,
            COL_ACC_Y: 0.0,
            COL_ACC_Z: 9.81,
            COL_ROT_W: sqrt_half,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: sqrt_half,
        }
        self.assert_zero_linear_acc(row)

    def test_inverted_acceleration(self):
        row = {
            COL_ACC_X: 0.0,
            COL_ACC_Y: 0.0,
            COL_ACC_Z: -9.81,
            COL_ROT_W: 1.0,
            COL_ROT_X: 0.0,
            COL_ROT_Y: 0.0,
            COL_ROT_Z: 0.0,
        }
        result = get_linear_acceleration(row)
        np.testing.assert_allclose(result, [0.0, 0.0, -19.62], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
