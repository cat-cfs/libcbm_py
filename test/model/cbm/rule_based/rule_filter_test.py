import unittest
from types import SimpleNamespace
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import rule_filter


class RuleFilterTest(unittest.TestCase):

    def test_merge_filters_error_on_intersecting_variables(self):
        """check if any 2 filters have intersecting values in local_dict
        property
        """
        a = SimpleNamespace(expression="expression", local_dict={"a": 1})
        b = SimpleNamespace(expression="expression", local_dict={"b": 1})
        c = SimpleNamespace(expression="expression", local_dict={"a": 1})
        with self.assertRaises(ValueError):
            rule_filter.merge_filters(a, b, c)

    def test_merge_filters_expected_output(self):
        """check the expected output of the merge_filters method
        """
        a = SimpleNamespace(expression="a == 1", local_dict={"a": 1})
        b = SimpleNamespace(expression="b == 2", local_dict={"b": 2})
        c = SimpleNamespace(expression="c == 3", local_dict={"c": 3})
        result = rule_filter.merge_filters(a, b, c)
        self.assertTrue(result.expression == "(a == 1) & (b == 2) & (c == 3)")
        self.assertTrue(result.local_dict == {"a": 1, "b": 2, "c": 3})

    def test_create_filter_expected_output(self):
        result = rule_filter.create_filter(
            "(A < 5) | (B > 6)",
            pd.DataFrame({
                "a": range(1, 10),
                "b": range(1, 10),
                "c": range(1, 10)}),
            columns=["a", "b"],
            column_variable_map={"a": "A", "b": "B"})

        self.assertTrue(result.expression == "(A < 5) | (B > 6)")
        self.assertTrue(list(result.local_dict["A"]) == list(range(1, 10)))
        self.assertTrue(list(result.local_dict["B"]) == list(range(1, 10)))

    def test_evaluate_filter_expected_output(self):
        result = rule_filter.evaluate_filter(
            rule_filter.create_filter(
                "(A < 5) | (B > 6)",
                pd.DataFrame({
                    "a": range(1, 10),
                    "b": range(1, 10),
                    "c": range(1, 10)}),
                columns=["a", "b"],
                column_variable_map={"a": "A", "b": "B"}))
        self.assertTrue(list(result) == [True]*4 + [False]*2 + [True]*3)

    def test_error_on_invalid_expression_evaluate_filter(self):
        with self.assertRaises(KeyError):
            rule_filter.evaluate_filter(
                rule_filter.create_filter(
                    "(A < 5) | (C > 6)",
                    pd.DataFrame({
                        "a": range(1, 10),
                        "b": range(1, 10),
                        "c": range(1, 10)}),
                    columns=["a", "b"],
                    column_variable_map={"a": "A", "b": "B"}))
