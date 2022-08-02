import unittest
import json
import numpy as np
from libcbm.storage import dataframe
from libcbm import resources
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper


TEST_CONFIG = {
    "pools": [
        {"id": 1, "index": 0, "name": "pool_1"},
        {"id": 2, "index": 1, "name": "pool_2"},
    ],
    "flux_indicators": [
        {
            "id": 1,
            "index": 0,
            "process_id": 1,
            "source_pools": [1],
            "sink_pools": [1],
        }
    ],
}


class WrapperTest(unittest.TestCase):
    def test_initialization_error(self):
        with self.assertRaises(RuntimeError):
            LibCBMHandle(resources.get_libcbm_bin_path(), "")

    def test_disposal(self):
        handle = LibCBMHandle(
            resources.get_libcbm_bin_path(), json.dumps(TEST_CONFIG)
        )
        with handle:
            self.assertTrue(handle.pointer > 0)
        self.assertTrue(handle.pointer == 0)

    def test_error_returned_by_library(self):
        handle = LibCBMHandle(
            resources.get_libcbm_bin_path(), json.dumps(TEST_CONFIG)
        )
        with self.assertRaises(RuntimeError):
            # try to free an unallocated op to trigger an error
            handle.call("LibCBM_Free_Op", 1)

    def test_enabled_supports_booleans_and_int32(self):
        handle = LibCBMHandle(
            resources.get_libcbm_bin_path(), json.dumps(TEST_CONFIG)
        )
        with handle:
            wrapper = LibCBMWrapper(handle)
            op = wrapper.allocate_op(2)
            wrapper.set_op(
                op,
                [np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0.5]])],
                np.array([0, 0], dtype="uintp"),
            )
            op_processes = [1]
            pools = dataframe.from_numpy({"pool_1": np.array([1.0, 1.0])})
            flux = dataframe.from_numpy({"f1": np.array([0.0, 0.0])})
            wrapper.compute_flux([op], op_processes, pools, flux)
