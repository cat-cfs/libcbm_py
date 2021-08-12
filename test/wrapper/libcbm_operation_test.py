import unittest

import numpy as np
from libcbm.wrapper import libcbm_operation
from test.wrapper import pool_flux_helpers


class LibCBMOperationTest(unittest.TestCase):

    def test_integration(self):
        pool_dict = {"a": 0, "b": 1, "c": 2}
        pooldef = pool_flux_helpers.create_pools(list(pool_dict.keys()))
        dll = pool_flux_helpers.load_dll({
            "pools": pooldef,
            "flux_indicators": []
        })

        op = libcbm_operation.Operation(
            dll, libcbm_operation.OperationFormat.RepeatingCoordinates,
            data=[
                [pool_dict["a"], pool_dict["a"], np.array([1., 1., 1.])],
                [pool_dict["a"], pool_dict["b"], np.array([2., 3., 4.])],
                [pool_dict["b"], pool_dict["b"], np.array([1., 1., 1.])],
                [pool_dict["c"], pool_dict["c"], np.array([1., 1., 1.])]
            ])

        pools_orig = np.ones(shape=(4, len(pool_dict)))
        pools_out = pools_orig.copy()
        op.set_matrix_index(np.array([0, 1, 2, 0], dtype=np.uint64))
        libcbm_operation.compute(dll, pools_out, [op])

        self.assertTrue(
            (pools_out == np.array([
                [1., 3., 1.],
                [1., 4., 1.],
                [1., 5., 1.],
                [1., 3., 1.]
            ])).all())
