import unittest
import pandas as pd
from types import SimpleNamespace
from libcbm.model.cbm import cbm_model


class ComputeProductionTest(unittest.TestCase):

    def test_compute_disturbance_production_expected_result(self):

        mock_pools = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1, 2, 3]})
        mock_inventory = pd.DataFrame({
            "age": [1, 1, 1],
            "area": [10, 20, 30]})
        flux_indicator_codes = [
            "DisturbanceSoftProduction", "DisturbanceHardProduction",
            "DisturbanceDOMProduction"]
        mock_flux = pd.DataFrame(
            data=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            columns=flux_indicator_codes)
        mock_eligible = pd.Series([True, True, True])
        mock_disturbance_type = 15

        model_functions = SimpleNamespace()

        def mock_get_disturbance_ops(op, inventory, parameters):
            self.assertTrue(op == 999)
            self.assertTrue(inventory.equals(mock_inventory))
            self.assertTrue(
                (parameters.disturbance_type ==
                 mock_disturbance_type).all())

        model_functions.get_disturbance_ops = mock_get_disturbance_ops

        compute_functions = SimpleNamespace()

        def mock_allocate_op(n_stands):
            self.assertTrue(n_stands == 3)
            return 999
        compute_functions.allocate_op = mock_allocate_op

        def mock_compute_flux(ops, op_processes, pools, flux, enabled):
            self.assertTrue(op_processes == [
                cbm_model.get_op_processes()["disturbance"]])
            self.assertTrue(ops == [999])
            self.assertTrue((pools == mock_pools.values).all())
            self.assertTrue(list(enabled) == list(mock_eligible))
            flux[:] = 1
        compute_functions.compute_flux = mock_compute_flux

        def mock_free_op(op):
            self.assertTrue(op == 999)

        mock_cbm_vars = SimpleNamespace(
            pools=mock_pools,
            flux=mock_flux,
            inventory=mock_inventory)
        compute_functions.free_op = mock_free_op
        cbm = cbm_model.CBM(
            compute_functions, model_functions, list(mock_pools.columns),
            flux_indicator_codes)
        result = cbm.compute_disturbance_production(
            mock_cbm_vars, mock_disturbance_type, mock_eligible)
        for flux_code in flux_indicator_codes:
            self.assertTrue(list(result[flux_code]) == [1, 1, 1])
        self.assertTrue(list(result["Total"]) == [3, 3, 3])
