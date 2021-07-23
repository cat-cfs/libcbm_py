import unittest
import numpy as np
from libcbm.model.moss_c import model_functions


class ModelFunctionsTest(unittest.TestCase):

    def test_expand_matrix(self):
        result = model_functions.expand_matrix(mat=[
            [1,  2,   np.array([3,   4,  5])],
            [6,  7,   np.array([8,   9, 10])],
            [11, 12,  np.array([13, 14, 15])],
        ], identity_set={})

        expected_output = [
            np.array([
                [1, 2, 3],
                [6, 7, 8],
                [11, 12, 13]], dtype=float),
            np.array([
                [1, 2, 4],
                [6, 7, 9],
                [11, 12, 14]], dtype=float),
            np.array([
                [1, 2, 5],
                [6, 7, 10],
                [11, 12, 15]], dtype=float),
        ]
        for i_mat, mat in enumerate(result):
            self.assertTrue((mat == expected_output[i_mat]).all())

    def test_expand_matrix_identity_set(self):
        result = model_functions.expand_matrix(mat=[
            [1,  1,   np.array([3,   4,  5])],
            [6,  7,   np.array([8,   9, 10])],
            [11, 12,  np.array([13, 14, 15])],
        ], identity_set={1, 2, 3})

        expected_output = [
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 3],
                [6, 7, 8],
                [11, 12, 13]], dtype=float),
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 4],
                [6, 7, 9],
                [11, 12, 14]], dtype=float),
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 5],
                [6, 7, 10],
                [11, 12, 15]], dtype=float),
        ]
        for i_mat, mat in enumerate(result):
            self.assertTrue((mat == expected_output[i_mat]).all())
