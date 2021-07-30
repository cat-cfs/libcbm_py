import unittest
from types import SimpleNamespace
import mock
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm import cbm_model


class RuleTargetTest(unittest.TestCase):

    def test_sorted_disturbance_target_error_on_less_than_zero_target(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([10, 1000, 0]),
                sort_var=pd.Series([1, 2, 3]),
                target=-10,
                eligible=pd.Series([True, True, True]))

    def test_sorted_disturbance_target_error_on_lt_zero_target_var(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([-1, 1000, 0]),
                sort_var=pd.Series([1, 2, 3]),
                target=10,
                eligible=pd.Series([True, True, True]))

    def test_sorted_disturbance_target_unrealized_on_zero_target_var_sum(self):

        rule_target.sorted_disturbance_target(
            target_var=pd.Series([0, 0, 0]),
            sort_var=pd.Series([1, 2, 3]),
            target=10,
            eligible=pd.Series([True, True, True]))

    def test_sorted_disturbance_target_on_unrealized_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=pd.Series([33, 33, 33]),
            sort_var=pd.Series([1, 2, 3]),
            target=100,
            eligible=pd.Series([True, True, True]))
        self.assertTrue(list(result.target.disturbed_index) == [2, 1, 0])
        self.assertTrue(
            list(result.target.area_proportions) == [1.0, 1.0, 1.0])
        self.assertTrue(list(result.target.target_var) == [33, 33, 33])
        self.assertTrue(list(result.target.sort_var) == [3, 2, 1])

    def test_sorted_disturbance_target_expected_result_with_exact_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=pd.Series([33, 33, 33]),
            sort_var=pd.Series([1, 2, 3]),
            target=99,
            eligible=pd.Series([True, True, True]))
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(list(result.target.disturbed_index) == [2, 1, 0])
        self.assertTrue(
            list(result.target.area_proportions) == [1.0, 1.0, 1.0])
        self.assertTrue(list(result.target.target_var) == [33, 33, 33])
        self.assertTrue(list(result.target.sort_var) == [3, 2, 1])

    def test_sorted_disturbance_target_expected_result_with_less_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=pd.Series([33, 33, 33]),
            sort_var=pd.Series([1, 2, 3]),
            target=34,
            eligible=pd.Series([True, True, True]))
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(list(result.target.disturbed_index) == [2, 1])
        self.assertTrue(list(result.target.area_proportions) == [1.0, 1/33])
        self.assertTrue(list(result.target.target_var) == [33, 33])
        self.assertTrue(list(result.target.sort_var) == [3, 2])

    def test_sorted_area_target_expected_result(self):

        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [1.5, 2.0, 2.0, 3.0]
        })
        result = rule_target.sorted_area_target(
            area_target_value=5.1,
            sort_value=mock_inventory.age,
            inventory=mock_inventory,
            eligible=pd.Series([True, True, True, True]))
        self.assertTrue(list(result.target.disturbed_index) == [3, 1, 2])
        self.assertTrue(list(result.target.target_var) == [3.0, 2.0, 2.0])
        self.assertTrue(list(result.target.sort_var) == [30, 20, 10])
        self.assertTrue(
            np.allclose(result.target.area_proportions, [1.0, 1.0, 0.1/2.0]))

    def test_sorted_area_target_error_on_dimension_mismatch(self):

        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [1.5, 2.0, 2.0, 3.0]
        })
        # note only 3 sort values, with 4 inventory rows
        with self.assertRaises(ValueError):
            rule_target.sorted_area_target(
                area_target_value=5.1,
                sort_value=pd.Series([1, 2, 3]),
                inventory=mock_inventory,
                eligible=pd.Series([True, True, True, True]))

    def test_sorted_merch_target_expected_result(self):
        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [2.0, 2.0, 2.0, 2.0]
        })
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10]})  # tonnes C/ha
        # since C targets are accumulated on mass values
        # the total production values here are actually
        # 20,20,20,20 tonnes using the above area multipliers

        result = rule_target.sorted_merch_target(
            carbon_target=55,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=pd.Series([4, 3, 2, 1]),
            efficiency=1.0,
            eligible=pd.Series([True, True, True, True]))
        self.assertTrue(list(result.target.disturbed_index) == [0, 1, 2])
        self.assertTrue(list(result.target.target_var) == [20, 20, 20])
        self.assertTrue(list(result.target.sort_var) == [4, 3, 2])
        self.assertTrue(
            np.allclose(result.target.area_proportions, [1.0, 1.0, 15/20]))

    def test_sorted_merch_target_expected_result_unrealized(self):
        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [2.0, 2.0, 2.0, 2.0]
        })
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10]})  # tonnes C/ha
        # since C targets are accumulated on mass values
        # the total production values here are actually
        # 20,20,20,20 tonnes using the above area multipliers

        # since the last index is not eligible maning the total production
        # available is 60 tonnes, and the target is 65
        result = rule_target.sorted_merch_target(
            carbon_target=65,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=pd.Series([4, 3, 2, 1]),
            efficiency=1.0,
            eligible=pd.Series([True, True, True, False]))  # note ineligible

        self.assertTrue(list(result.target.disturbed_index) == [0, 1, 2])
        self.assertTrue(list(result.target.target_var) == [20, 20, 20])
        self.assertTrue(list(result.target.sort_var) == [4, 3, 2])
        self.assertTrue(
            np.allclose(result.target.area_proportions, [1.0, 1.0, 1.0]))

    def test_sorted_merch_target_error_on_dimension_mismatch1(self):

        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [2.0, 2.0, 2.0, 2.0]
        })

        # note 5 values here, and 4 in inventory
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10, 10]})

        with self.assertRaises(ValueError):
            rule_target.sorted_merch_target(
                carbon_target=55,
                disturbance_production=mock_disturbance_production,
                inventory=mock_inventory,
                sort_value=pd.Series([4, 3, 2, 1]),
                efficiency=1.0,
                eligible=pd.Series([True, True, True, True]))

    def test_sorted_merch_target_error_on_dimension_mismatch2(self):

        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [2.0, 2.0, 2.0, 2.0]
        })

        # note 5 values here, and 4 in inventory
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10]})

        with self.assertRaises(ValueError):
            rule_target.sorted_merch_target(
                carbon_target=55,
                disturbance_production=mock_disturbance_production,
                inventory=mock_inventory,
                sort_value=pd.Series([4, 3, 2, 1, 15]),  # extra here
                efficiency=1.0,
                eligible=pd.Series([True, True, True, True]))

    def test_sorted_merch_target_expected_result_with_efficiency(self):
        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [1.0, 2.0, 1.0, 1.0]
        })
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10]})
        # with efficiency < 1.0 the disturbance production is lowered,
        # and all records will be split

        result = rule_target.sorted_merch_target(
            carbon_target=33,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=pd.Series([4, 3, 2, 1]),
            efficiency=0.8,
            eligible=pd.Series([True, True, True, True]))
        self.assertTrue(list(result.target.disturbed_index) == [0, 1, 2, 3])

        # efficiency*production causes this
        self.assertTrue(list(result.target.target_var) == [8, 16, 8, 8])

        self.assertTrue(list(result.target.sort_var) == [4, 3, 2, 1])

        # (0.8 * 10 + 0.8 * 20 + 0.8 * 10) == 32
        # (10 * x) == 33 - 32 == 1
        # x = 1/10

        # carbon_target = 0.8 * 3 * 10 + 1/10 = 25
        self.assertTrue(
            np.allclose(result.target.area_proportions, [0.8, 0.8, 0.8, 1/10]))

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

        def mock_get_disturbance_ops(op, inventory, disturbance_type):
            self.assertTrue(op == 999)
            self.assertTrue(inventory.equals(mock_inventory))
            self.assertTrue(
                (disturbance_type == mock_disturbance_type).all()[0])

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
            flux_indicators=mock_flux,
            inventory=mock_inventory)
        compute_functions.free_op = mock_free_op
        result = rule_target.compute_disturbance_production(
            model_functions, compute_functions, mock_cbm_vars,
            mock_disturbance_type, mock_eligible)
        for flux_code in flux_indicator_codes:
            self.assertTrue(list(result[flux_code]) == [1, 1, 1])
        self.assertTrue(list(result["Total"]) == [3, 3, 3])

    def test_proportion_area_target(self):
        mock_inventory = pd.DataFrame({
            "area": [60.0, 60.0, 60.0, 60.0]
        })
        result = rule_target.proportion_area_target(
            area_target_value=100,
            inventory=mock_inventory,
            eligible=pd.Series([True, False, True, False]))
        self.assertTrue(
            list(result.target.target_var) == [5 / 6 * 60, 5 / 6 * 60])
        self.assertTrue(result.target.sort_var is None)
        self.assertTrue(list(result.target.disturbed_index) == [0, 2])
        self.assertTrue(list(result.target.area_proportions) == [5 / 6, 5 / 6])

    def test_proportion_merch_target(self):
        mock_inventory = pd.DataFrame({
            "area": [12.0, 11.0, 10.0, 7.0],
        })
        carbon_target = 100.0
        disturbance_production = pd.Series([5.0, 5.0, 4.0, 10.0])
        eligible = pd.Series([True, False, True, True])
        efficiency = 0.9
        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=efficiency,
            eligible=eligible)

        # the sequence of eligible disturbance production is:
        # (area*production*efficiency)
        # 12.0 * 5.0 * 0.9 + 10.0 * 4.0 * 0.9 + 7.0 * 10.0 * 0.9
        # sum of the above is 54 + 36 + 63 = 153 (tC eligible production)
        # area proportion = 100/153 = 0.65359477

        self.assertTrue(
            list(result.target.target_var) ==
            [12.0 * 5.0 * 0.9 * 100 / 153,
             10.0 * 4.0 * 0.9 * 100 / 153,
             7.0 * 10.0 * 0.9 * 100 / 153])
        self.assertTrue(result.target.sort_var is None)
        self.assertTrue(list(result.target.disturbed_index) == [0, 2, 3])
        self.assertTrue(
            list(result.target.area_proportions) == [
                100 / 153 * 0.9, 100 / 153 * 0.9, 100 / 153 * 0.9
            ])

    def test_proportion_sort_proportion_target(self):
        mock_inventory = pd.DataFrame({
            "area": [10, 20, 30, 40, 50, 60]
        })
        eligible = pd.Series([True, False, True, True, False, True])
        proportion_target = 3/4

        result = rule_target.proportion_sort_proportion_target(
            proportion_target=proportion_target,
            inventory=mock_inventory,
            eligible=eligible)

        self.assertTrue(
            list(result.target.target_var) == [
                3 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4])
        self.assertTrue(result.target.sort_var is None)
        self.assertTrue(list(result.target.disturbed_index) == [0, 2, 3, 4])
        self.assertTrue(
            list(result.target.area_proportions) == [
                3 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4])
