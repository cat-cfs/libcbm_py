import unittest
from unittest.mock import patch
from mock import Mock
import pandas as pd
import numpy as np
from types import SimpleNamespace
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

        tr_group_key = {"disturbance_type_id": 10, "a": "?"}
        tr_group = pd.DataFrame()
        transition_mask = np.array([True, True], dtype=bool)
        mock_cbm_vars = SimpleNamespace(
            classifiers=pd.DataFrame({"a": [1, 2, 3]}),
            inventory="mock_inventory",
            pools="mock_pools",
            state="mock_state_variables",
            params=pd.DataFrame({"disturbance_type": [0, 0, 0]}))
        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            # since the mocked rule filter returns an array that has True at
            # index 0 an error should be raised, since the transition_mask also
            # has true at index 0
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True, False], dtype=bool)
            with self.assertRaises(ValueError):
                tr_processor.apply_transition_rule(
                    tr_group_key, tr_group, transition_mask, mock_cbm_vars)

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
        mock_classifiers = pd.DataFrame({
            "a": [1],
            "b": [3]
        })
        mock_inventory = pd.DataFrame({
            "area": [1.0]
        })
        mock_pools = pd.DataFrame({
            "p0": [1],
            "p1": [4]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0],
            "regeneration_delay": [0]
        })
        mock_params = pd.DataFrame({
            "disturbance_type": [0],
            "reset_age": [-1]})
        mock_flux = pd.DataFrame({
            "f1": [0],
            "f2": [0]
        })
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            params=mock_params,
            flux=mock_flux)

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True], dtype=bool)
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                 tr_group_key, tr_group, transition_mask, mock_cbm_vars)
            self.assertTrue(list(transition_mask) == [True])
            self.assertTrue(list(cbm_vars.state.regeneration_delay) == [10])
            self.assertTrue(list(cbm_vars.state.age) == [0])
            self.assertTrue(list(cbm_vars.params.reset_age) == [40])

            # note the changed classifier id
            self.assertTrue(list(cbm_vars.classifiers.a) == [2])

            # note unchanged since "?" was used for transition classifier value
            self.assertTrue(list(cbm_vars.classifiers.b) == [3])

            self.assertTrue(list(cbm_vars.inventory.area) == [1.0])
            self.assertTrue(list(cbm_vars.pools.p0) == [1])
            self.assertTrue(list(cbm_vars.pools.p1) == [4])
            self.assertTrue(list(cbm_vars.flux.f1) == [0])
            self.assertTrue(list(cbm_vars.flux.f2) == [0])

    def test_single_record_split_remainder_transition(self):
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
            "percent": [50]
        })
        transition_mask = np.array([False], dtype=bool)
        mock_classifiers = pd.DataFrame({
            "a": [1],
            "b": [3]
        })
        mock_inventory = pd.DataFrame({
            "area": [1.0]
        })
        mock_pools = pd.DataFrame({
            "p0": [1],
            "p1": [4]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0],
            "regeneration_delay": [0]
        })
        mock_params = pd.DataFrame({
            "disturbance_type": [0],
            "reset_age": [-1]})
        mock_flux = pd.DataFrame({
            "f1": [0],
            "f2": [0]
        })
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            params=mock_params,
            flux_indicators=mock_flux)

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True], dtype=bool)
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                 tr_group_key, tr_group, transition_mask, mock_cbm_vars)

            self.assertTrue(list(transition_mask) == [True, True])
            self.assertTrue(list(cbm_vars.state.regeneration_delay) == [10, 0])
            self.assertTrue(list(cbm_vars.state.age) == [0, 0])
            self.assertTrue(list(cbm_vars.params.reset_age) == [40, -1])

            # note the changed classifier id for index 0, but not for index 1
            # (because of the remainder split)
            self.assertTrue(list(cbm_vars.classifiers.a) == [2, 1])

            # note unchanged since "?" was used for transition classifier value
            self.assertTrue(list(cbm_vars.classifiers.b) == [3, 3])

            self.assertTrue(list(cbm_vars.inventory.area) == [0.5, 0.5])

            # since pools are area densities the are just copied here
            self.assertTrue(list(cbm_vars.pools.p0) == [1, 1])
            self.assertTrue(list(cbm_vars.pools.p1) == [4, 4])

            self.assertTrue(list(cbm_vars.flux_indicators.f1) == [0, 0])
            self.assertTrue(list(cbm_vars.flux_indicators.f2) == [0, 0])
