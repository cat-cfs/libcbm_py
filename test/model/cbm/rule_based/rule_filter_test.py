import unittest
from types import SimpleNamespace
import pandas as pd
from libcbm.model.cbm.rule_based import rule_filter


class RuleFilterTest(unittest.TestCase):

    def test_evaluate_filter_none_result(self):
        mock_filter = SimpleNamespace()
        mock_filter.expression = None
        self.assertTrue(rule_filter.evaluate_filters(mock_filter) is None)
        self.assertTrue(rule_filter.evaluate_filters(None) is None)

    def test_create_filter_expected_output(self):
        result = rule_filter.create_filter(
            "(A < 5) | (B > 6)",
            pd.DataFrame({
                "A": range(1, 10),
                "B": range(1, 10),
                "C": range(1, 10)}))

        self.assertTrue(result.expression == "(A < 5) | (B > 6)")
        self.assertTrue(list(result.local_dict["A"]) == list(range(1, 10)))
        self.assertTrue(list(result.local_dict["B"]) == list(range(1, 10)))

    def test_evaluate_filter_expected_output(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                "(a < 5) | (b > 6)",
                pd.DataFrame({
                    "a": range(1, 10),
                    "b": range(1, 10),
                    "c": range(1, 10)})))
        self.assertTrue(list(result) == [True]*4 + [False]*2 + [True]*3)

    def test_evaluate_filter_expected_output2(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                "a == 1",
                pd.DataFrame({"a": [0, 1, 0, 1]})),
            rule_filter.create_filter(
                "a == 1",
                pd.DataFrame({"a": [0, 0, 1, 1]})))

        self.assertTrue(list(result) == [False, False, False, True])

    def test_error_on_invalid_expression_evaluate_filter(self):
        with self.assertRaises(KeyError):
            rule_filter.evaluate_filters(
                rule_filter.create_filter(
                    "(A < 5) | (C > 6)",
                    pd.DataFrame({
                        "a": range(1, 10),
                        "c": range(1, 10)})))
