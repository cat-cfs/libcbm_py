import unittest

import numpy as np
from libcbm.wrapper import libcbm_operation
from libcbm.storage import dataframe
from test.wrapper import pool_flux_helpers


class LibCBMOperationTest(unittest.TestCase):
    def test_integration(self):
        pool_dict = {"a": 0, "b": 1, "c": 2}
        pooldef = pool_flux_helpers.create_pools(list(pool_dict.keys()))
        dll = pool_flux_helpers.load_dll(
            {"pools": pooldef, "flux_indicators": []}
        )

        op = libcbm_operation.Operation(
            dll,
            libcbm_operation.OperationFormat.RepeatingCoordinates,
            data=[
                [pool_dict["a"], pool_dict["a"], np.array([1.0, 1.0, 1.0])],
                [pool_dict["a"], pool_dict["b"], np.array([2.0, 3.0, 4.0])],
                [pool_dict["b"], pool_dict["b"], np.array([1.0, 1.0, 1.0])],
                [pool_dict["c"], pool_dict["c"], np.array([1.0, 1.0, 1.0])],
            ],
            op_process_id=0,
        )

        pools_orig = np.ones(shape=(4, len(pool_dict)))
        pools_out = dataframe.from_numpy(
            {name: pools_orig[:, idx] for name, idx in pool_dict.items()}
        )
        op.set_op(np.array([0, 1, 2, 0], dtype=np.uint64))
        libcbm_operation.compute(dll, pools_out, [op])

        self.assertTrue(
            (
                pools_out.to_numpy()
                == np.array(
                    [
                        [1.0, 3.0, 1.0],
                        [1.0, 4.0, 1.0],
                        [1.0, 5.0, 1.0],
                        [1.0, 3.0, 1.0],
                    ]
                )
            ).all()
        )
