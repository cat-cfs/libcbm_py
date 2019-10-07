import unittest
import pandas as pd
from libcbm.model.cbm.rule_based import rule_target


class RuleTargetTest(unittest.TestCase):

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