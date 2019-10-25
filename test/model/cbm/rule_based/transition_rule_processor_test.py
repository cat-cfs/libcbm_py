
import unittest
import pandas as pd
from libcbm.model.cbm.rule_based import transition_rule_processor


class TransitionRuleProcessorTest(unittest.TestCase):

    def test_create_split_proportions_percentage_error(self):

        mock_tr_group_key = {"a": 1, "b": 2}
        mock_tr_group = pd.DataFrame({"percent": [50, 51]})
        with self.assertRaises(ValueError):
            transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=mock_tr_group,
                group_error_max=1)

    def test_create_split_proportions_with_100_percent(self):
        mock_tr_group_key = {"a": 1, "b": 2}
        self.assertTrue(
            list(transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=pd.DataFrame({"percent": [100.01]}),
                group_error_max=0.1)) == [1.0])
        self.assertTrue(
            list(transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=pd.DataFrame({"percent": [99.99]}),
                group_error_max=0.1)) == [1.0])
        self.assertTrue(
            list(transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=pd.DataFrame({"percent": [50, 50]}),
                group_error_max=0.1)) == [0.5, 0.5])

    def test_create_split_proportions_with_less_than_100_percent(self):
        mock_tr_group_key = {"a": 1, "b": 2}
        self.assertTrue(
            list(transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=pd.DataFrame({"percent": [85]}),
                group_error_max=0.1)) == [0.85, 0.15])
        self.assertTrue(
            list(transition_rule_processor.create_split_proportions(
                tr_group_key=mock_tr_group_key,
                tr_group=pd.DataFrame({"percent": [45, 35]}),
                group_error_max=0.1)) == [0.45, 0.35, 0.20])
