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

        tr_group_key = {"disturbance_type_id": 10}
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

    def test_single_record_transition(self):
        mock_classifier_filter_builder = Mock()
        mock_state_variable_filter_func = Mock()
        mock_classifier_config = {
            "classifiers": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"}]}
        grouped_percent_err_max = 0.001
        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_filter_builder, mock_state_variable_filter_func,
            mock_classifier_config, grouped_percent_err_max, wildcard,
            transition_classifier_postfix)

        tr_group_key = {
            "a": "a1", "b": "?", "disturbance_type_id": 55}
        tr_group = pd.DataFrame({
            "a": ["a1"],
            "b": ["b1"],
            "disturbance_type_id": [55],
            "a_tr": ["a2"],
            "b_tr": ["?"],
            "regeneration_delay": [10],
            "reset_age": [40],
            "percent": [100]
        })
        transition_mask = np.array([False], dtype=bool)
        mock_disturbance_type = np.array([55])
        mock_classifiers = pd.DataFrame({
            "a": [1],
            "b": [3]
        })
        mock_inventory = pd.DataFrame({
            "area": [1.0]
        })
        mock_pools = pd.DataFrame({
            "p0": [1],
            "p1": [1]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0],
        })
        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True], dtype=bool)
            (transition_mask,
             transition_output,
             classifiers,
             inventory,
             pools,
             state_variables) = tr_processor.apply_transition_rule(
                 tr_group_key, tr_group, transition_mask,
                 mock_disturbance_type, mock_classifiers, mock_inventory,
                 mock_pools, mock_state_variables)
            self.assertTrue(list(transition_mask) == [True])
            self.assertTrue(list(transition_output.regeneration_delay) == [10])
            self.assertTrue(list(transition_output.reset_age) == [40])

            # note the changed classifier id
            self.assertTrue(list(classifiers.a) == [2])

            # note unchanged since "?" was used for transition classifier value
            self.assertTrue(list(classifiers.b) == [3])

            self.assertTrue(inventory.equals(mock_inventory))
            self.assertTrue(pools.equals(mock_pools))
            self.assertTrue(state_variables.equals(state_variables))

    def test_single_record_split_remainder_transition(self):
        return # TODO continue refactor
        mock_classifier_filter_builder = Mock()
        mock_state_variable_filter_func = Mock()
        mock_classifier_config = {
            "classifiers": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"}]}
        grouped_percent_err_max = 0.001
        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_filter_builder, mock_state_variable_filter_func,
            mock_classifier_config, grouped_percent_err_max, wildcard,
            transition_classifier_postfix)

        tr_group_key = {
            "a": "a1", "b": "?", "disturbance_type_id": 55}
        tr_group = pd.DataFrame({
            "a": ["a1"],
            "b": ["b1"],
            "disturbance_type_id": [55],
            "a_tr": ["a2"],
            "b_tr": ["?"],
            "regeneration_delay": [7],
            "reset_age": [4],
            "percent": [35.0]
            # since the group's sum is less than 100, the inventory
            # will be split in half, and the remainder is not transitioned
        })
        transition_mask = np.array([False], dtype=bool)
        mock_disturbance_type = np.array([55])
        mock_classifiers = pd.DataFrame({
            "a": [1],
            "b": [3]
        })
        mock_inventory = pd.DataFrame({
            "area": [1.0]
        })
        mock_pools = pd.DataFrame({
            "p0": [1],
            "p1": [1]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0],
        })
        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True], dtype=bool)
            (transition_mask,
             transition_output,
             classifiers,
             inventory,
             pools,
             state_variables) = tr_processor.apply_transition_rule(
                 tr_group_key, tr_group, transition_mask,
                 mock_disturbance_type, mock_classifiers, mock_inventory,
                 mock_pools, mock_state_variables)
            self.assertTrue(list(transition_mask) == [True, False])
            self.assertTrue(
                list(transition_output.regeneration_delay) == [7, 0])
            self.assertTrue(list(transition_output.reset_age) == [4, -1])

            # the transition portion is changed
            self.assertTrue(list(classifiers.a) == [2, 1])

            self.assertTrue(list(classifiers.b) == [3, 3])

            self.assertTrue(list(inventory.area) == [35.0, 65.0])
            self.assertTrue(pools.iloc[0].equals(mock_pools))
            self.assertTrue(pools.iloc[1].equals(mock_pools))

            self.assertTrue(state_variables.iloc[0].equals(state_variables))
            self.assertTrue(state_variables.iloc[1].equals(state_variables))


