import unittest
from unittest.mock import patch
from mock import Mock
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import transition_rule_processor
from libcbm.model.cbm.rule_based.transition_rule_processor \
    import TransitionRuleProcessor

PATCH_PATH = "libcbm.model.cbm.rule_based.transition_rule_processor"


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

    def test_overlapping_transition_rules_error(self):
        mock_classifier_filter_builder = Mock()
        mock_state_variable_filter_func = Mock()
        mock_classifier_config = {
            "classifiers": [{"id": 1, "name": "a"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"}]}
        grouped_percent_err_max = 0.001
        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_filter_builder, mock_state_variable_filter_func,
            mock_classifier_config, grouped_percent_err_max, wildcard,
            transition_classifier_postfix)

        tr_group_key = {"disturbance_type": 10}
        tr_group = pd.DataFrame()
        transition_mask = np.array([True, True], dtype=bool)
        disturbance_type = np.ones(2)
        classifiers = pd.DataFrame()
        inventory = pd.DataFrame()
        pools = pd.DataFrame()
        state_variables = pd.DataFrame()
        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            # since the mocked rule filter returns an array that has True at
            # index 0 an error should be raised, since the transition_mask also
            # has true at index 0
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True, False], dtype=bool)
            with self.assertRaises(ValueError):
                tr_processor.apply_transition_rule(
                    tr_group_key, tr_group, transition_mask, disturbance_type,
                    classifiers, inventory, pools, state_variables)
