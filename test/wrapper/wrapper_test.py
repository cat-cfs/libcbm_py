import unittest
import json

from libcbm import resources
from libcbm.wrapper.libcbm_handle import LibCBMHandle


TEST_CONFIG = {
    "pools": [{"id": 1, "index": 0, "name": "pool_1"}],
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
