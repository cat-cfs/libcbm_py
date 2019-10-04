import unittest
from types import SimpleNamespace
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import rule_filter


class RuleFilterTest(unittest.TestCase):

    def test_error_on_intersecting_variables(self):
        """check if any 2 filters have intersecting values in local_dict
        property
        """
        a = SimpleNamespace(expression="expression", local_dict={"a": 1})
        b = SimpleNamespace(expression="expression", local_dict={"b": 1})
        c = SimpleNamespace(expression="expression", local_dict={"a": 1})
        with self.assertRaises(ValueError):
            rule_filter.merge_filters(a, b, c)
