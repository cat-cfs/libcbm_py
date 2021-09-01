import unittest
import numpy as np
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int


class LibCBM_Matrix_Test(unittest.TestCase):

    def test_1_by_1_matrix(self):
        for arr in [np.array(1.0), np.array([1.0]), np.array([[[1.0]]]),
                    np.ones(shape=((1, 1, 1,)))]:
            mat = LibCBM_Matrix(arr)
            self.assertTrue(mat.rows == 1)
            self.assertTrue(mat.cols == 1)

    def test_incorrect_dimensions_matrix(self):
        with self.assertRaises(ValueError):
            LibCBM_Matrix(np.ones(shape=(1, 2, 3)))

    def test_incorrect_type_int(self):
        with self.assertRaises(ValueError):
            LibCBM_Matrix(np.ones(shape=(1, 1), dtype=int))

        with self.assertRaises(ValueError):
            LibCBM_Matrix_Int(np.ones(shape=(1, 1), dtype=float))

    def test_error_on_non_c_contiguous(self):
        with self.assertRaises(ValueError):
            arr = np.ones(shape=(3, 2), dtype=float, order="F")
            LibCBM_Matrix(arr)
        #np.ones(shape=((1, 1, 1,)))]:
