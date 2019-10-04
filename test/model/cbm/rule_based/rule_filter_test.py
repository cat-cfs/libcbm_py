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

