import unittest
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import rule_target


class RuleTargetTest(unittest.TestCase):

    def test_sorted_disturbance_target_error_on_less_than_zero_target(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([10, 1000, 0]),
                sort_var=pd.Series([1, 2, 3]),
                target=-10)

    def test_sorted_disturbance_target_error_on_lt_zero_target_var(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([-1, 1000, 0]),
                sort_var=pd.Series([1, 2, 3]),
                target=10)

    def test_sorted_disturbance_target_error_on_zero_target_var_sum(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([0, 0, 0]),
                sort_var=pd.Series([1, 2, 3]),
                target=10)

    def test_sorted_disturbance_target_error_on_unrealized_target(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=pd.Series([33, 33, 33]),
                sort_var=pd.Series([1, 2, 3]),
                target=100)

    def test_sorted_disturbance_target_expected_result_with_exact_target(self):
        result = rule_target.sorted_disturbance_target(
            target_var=pd.Series([33, 33, 33]),
            sort_var=pd.Series([1, 2, 3]),
            target=99)
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(list(result.disturbed_indices) == [2, 1, 0])
        self.assertTrue(list(result.area_proportions) == [1.0, 1.0, 1.0])

    def test_sorted_disturbance_target_expected_result_with_less_target(self):
        result = rule_target.sorted_disturbance_target(
            target_var=pd.Series([33, 33, 33]),
            sort_var=pd.Series([1, 2, 3]),
            target=34)
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(list(result.disturbed_indices) == [2, 1])
        self.assertTrue(list(result.area_proportions) == [1.0, 1/33])

    def test_sorted_area_target_expected_result(self):
        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [1.5, 2.0, 2.0, 3.0]
        })
        result = rule_target.sorted_area_target(
            area_target_value=5.1,
            sort_value=mock_inventory.age,
            inventory=mock_inventory)
        self.assertTrue(list(result.disturbed_indices) == [3, 1, 2])
        self.assertTrue(
            np.allclose(result.area_proportions, [1.0, 1.0, 0.1/2.0]))

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
            efficiency=1.0)
        self.assertTrue(list(result.disturbed_indices) == [0, 1, 2])
        self.assertTrue(
            np.allclose(result.area_proportions, [1.0, 1.0, 15/20]))

    def test_sorted_merch_target_expected_result_with_efficiency(self):
        mock_inventory = pd.DataFrame({
            "age": [0, 20, 10, 30],
            "area": [1.0, 1.0, 1.0, 1.0]
        })
        mock_disturbance_production = pd.DataFrame(
            {"Total": [10, 10, 10, 10]})
        # with efficiency < 1.0 the disturbance production is lowered,
        # and all records will be split

        result = rule_target.sorted_merch_target(
            carbon_target=25,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=pd.Series([4, 3, 2, 1]),
            efficiency=0.8)
        self.assertTrue(list(result.disturbed_indices) == [0, 1, 2, 3])

        # (0.8 * 10 + 0.8 * 10 + 0.8 * 10) == 24
        # (10 * x) == 1
        # x = 1/10

        # carbon_target = 0.8 * 3 * 10 + 1/10 = 25
        self.assertTrue(
            np.allclose(result.area_proportions, [0.8, 0.8, 0.8, 1/10]))
