import unittest
import numpy as np
from test.model.cbm import cbm3_simulation_results
from libcbm.test.cbm import state_comparison
from libcbm.test.cbm import pool_comparison
from libcbm.test.cbm import flux_comparison
from libcbm.test.cbm import result_comparison
from libcbm.test.cbm import test_case_simulator


class CBM3_Comparison(unittest.TestCase):
    """Tests of libcbm results versus cbm3 results
    """

    def compare_result(self, test_name):
        """Compare merged cbm/libcbm dataframes using numpy.allclose and
        fail on any instance where allclose is false

        Args:
            test_name (str): the name of the cbm3 test result to compare with
                libcbm.  This is used as a subdirectory name to fetch stored
                results.
        """
        cbm3_test = cbm3_simulation_results.get_results(test_name)
        libcbm_result = test_case_simulator.run_test_cases(
            cbm3_test.cases, cbm3_test.metadata["n_steps"])

        libcbm_result_suffix = result_comparison.get_libcbm_result_suffix()
        cbm3_result_suffix = result_comparison.get_cbm3_result_suffix()

        merged_state = state_comparison.get_merged_state(
            cbm3_test.state, libcbm_result["state"])

        for col in ["age", "land_class"]:
            self.assertTrue(
                np.allclose(
                    merged_state[f"{col}{libcbm_result_suffix}"],
                    merged_state[f"{col}{cbm3_result_suffix}"]))

        merged_pools = pool_comparison.get_merged_pools(
            cbm3_test.pools, libcbm_result["pools"])

        for pool in pool_comparison.get_libcbm_pools():
            libcbm_col = f"{pool}{libcbm_result_suffix}"
            cbm3_col = f"{pool}{cbm3_result_suffix}"
            self.assertTrue(
                np.allclose(merged_pools[libcbm_col], merged_pools[cbm3_col])
            )

    def test_basic_afforestation(self):
        """compare a single stand afforestation simulation
        """
        self.compare_result("basic_afforestation")
