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
            "b": ["?"],
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

    def test_single_record_split_transition(self):
        mock_classifier_filter_builder = Mock()
        mock_state_variable_filter_func = Mock()
        mock_classifier_config = {
            "classifiers": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"},
                {"id": 4, "classifier_id": 2, "value": "b2"}]}
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
            "a": ["a1", "a1"],
            "b": ["?", "?"],
            "disturbance_type_id": [55, 55],
            "a_tr": ["a2", "a1"],
            "b_tr": ["?", "b2"],
            "regeneration_delay": [10, -1],
            "reset_age": [40, 21],
            "percent": [35, 65]
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
            "p0": [33],
            "p1": [11]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0],
            "regeneration_delay": [0]
        })
        mock_params = pd.DataFrame({
            "disturbance_type": [0],
            "reset_age": [-1]})
        mock_flux = pd.DataFrame({
            "f1": [10],
            "f2": [100]
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
            self.assertTrue(
                list(cbm_vars.state.regeneration_delay) == [10, -1])
            self.assertTrue(list(cbm_vars.state.age) == [0, 0])
            self.assertTrue(list(cbm_vars.params.reset_age) == [40, 21])
            self.assertTrue(list(cbm_vars.classifiers.a) == [2, 1])
            self.assertTrue(list(cbm_vars.classifiers.b) == [3, 4])
            self.assertTrue(list(cbm_vars.inventory.area) == [0.35, 0.65])
            self.assertTrue(list(cbm_vars.pools.p0) == [33, 33])
            self.assertTrue(list(cbm_vars.pools.p1) == [11, 11])
            self.assertTrue(list(cbm_vars.flux_indicators.f1) == [10, 10])
            self.assertTrue(list(cbm_vars.flux_indicators.f2) == [100, 100])

    def test_multiple_records_multiple_split_transitions(self):
        mock_classifier_filter_builder = Mock()
        mock_state_variable_filter_func = Mock()
        mock_classifier_config = {
            "classifiers": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 1, "value": "a3"},
                {"id": 4, "classifier_id": 2, "value": "b1"},
                {"id": 5, "classifier_id": 2, "value": "b2"},
                {"id": 6, "classifier_id": 2, "value": "b3"}]}

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
            "a": ["a1", "a1", "a1"],
            "b": ["?", "?", "?"],
            "disturbance_type_id": [55, 55, 55],
            "a_tr": ["a2", "a1", "a3"],
            "b_tr": ["?", "b1", "b2"],
            "regeneration_delay": [1, 2, 3],
            "reset_age": [1, 2, 3],
            "percent": [10, 10, 10]
        })
        transition_mask = np.array([False], dtype=bool)
        mock_classifiers = pd.DataFrame({
            "a": [1, 2, 1],
            "b": [3, 6, 4]
        })
        mock_inventory = pd.DataFrame({
            "index": [0, 1, 2],
            "area": [1.0, 5.0, 10.0]
        })
        mock_pools = pd.DataFrame({
            "p0": [33, 22, 11],
            "p1": [11, 0, -11]
        })
        mock_state_variables = pd.DataFrame({
            "age": [0, 1, 2],
            "regeneration_delay": [0, 0, 0]
        })
        mock_params = pd.DataFrame({
            "disturbance_type": [0, 0, 0],
            "reset_age": [-1, -1, -1]})
        mock_flux = pd.DataFrame({
            "f1": [10, 20, 30],
            "f2": [100, 90, 80]
        })
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            params=mock_params,
            flux_indicators=mock_flux)

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            # for the test, indexes 0 and 2 will be eligible
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda filter_obj: np.array([True, False, True], dtype=bool)
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                 tr_group_key, tr_group, transition_mask, mock_cbm_vars)

            result = cbm_vars.inventory \
                .join(cbm_vars.classifiers) \
                .join(cbm_vars.state) \
                .join(cbm_vars.params) \
                .join(cbm_vars.pools) \
                .join(cbm_vars.flux_indicators) \
                .join(
                    pd.DataFrame(
                        {"transition_mask": transition_mask}))

            transitioned_1 = result[result["index"] == 0]
            non_transitioned = result[result["index"] == 1]
            transitioned_2 = result[result["index"] == 2]

            self.assertTrue(
                ((transitioned_1.f1 == 10) & (transitioned_1.f2 == 100) &
                 (transitioned_1.p0 == 33) & (transitioned_1.p1 == 11) &
                 (transitioned_1.transition_mask.all())).all())
            self.assertTrue(
                ((non_transitioned.f1 == 20) & (non_transitioned.f2 == 90) &
                 (non_transitioned.p0 == 22) & (non_transitioned.p1 == 0) &
                 (~non_transitioned.transition_mask.all())).all())
            self.assertTrue(
                ((transitioned_2.f1 == 30) & (transitioned_2.f2 == 80) &
                 (transitioned_2.p0 == 11) & (transitioned_2.p1 == -11) &
                 (transitioned_2.transition_mask.all())).all())

            self.assertTrue(transitioned_1.shape[0] == 4)
            self.assertTrue(transitioned_1.area.sum() == 1.0)

            self.assertTrue(non_transitioned.shape[0] == 1)
            self.assertTrue(non_transitioned.area.sum() == 5.0)

            self.assertTrue(transitioned_2.shape[0] == 4)
            self.assertTrue(transitioned_2.area.sum() == 10.0)

            transitioned_1_v1 = transitioned_1[
                transitioned_1.regeneration_delay == 1]
            self.assertTrue(transitioned_1_v1.area.sum() == 1.0/10.0)
            self.assertTrue(transitioned_1_v1.reset_age.sum() == 1)
            self.assertTrue(transitioned_1_v1.a.sum() == 2)
            self.assertTrue(transitioned_1_v1.b.sum() == 3)
            transitioned_1_v2 = transitioned_1[
                transitioned_1.regeneration_delay == 2]
            self.assertTrue(transitioned_1_v2.area.sum() == 1.0/10.0)
            self.assertTrue(transitioned_1_v2.reset_age.sum() == 2)
            self.assertTrue(transitioned_1_v2.a.sum() == 1)
            self.assertTrue(transitioned_1_v2.b.sum() == 4)
            transitioned_1_v3 = transitioned_1[
                transitioned_1.regeneration_delay == 3]
            self.assertTrue(transitioned_1_v3.area.sum() == 1.0/10.0)
            self.assertTrue(transitioned_1_v3.reset_age.sum() == 3)
            self.assertTrue(transitioned_1_v3.a.sum() == 3)
            self.assertTrue(transitioned_1_v3.b.sum() == 5)
            transitioned_1_v4 = transitioned_1[
                transitioned_1.regeneration_delay == 0]
            self.assertTrue(transitioned_1_v4.area.sum() == 1.0 - (3.0 / 10.0))
            self.assertTrue(transitioned_1_v4.reset_age.sum() == -1)
            self.assertTrue(transitioned_1_v4.a.sum() == 1)
            self.assertTrue(transitioned_1_v4.b.sum() == 3)

            transitioned_2_v1 = transitioned_2[
                transitioned_2.regeneration_delay == 1]
            self.assertTrue(transitioned_2_v1.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v1.reset_age.sum() == 1)
            self.assertTrue(transitioned_2_v1.a.sum() == 2)
            self.assertTrue(transitioned_2_v1.b.sum() == 4)
            transitioned_2_v2 = transitioned_2[
                transitioned_2.regeneration_delay == 2]
            self.assertTrue(transitioned_2_v2.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v2.reset_age.sum() == 2)
            self.assertTrue(transitioned_2_v2.a.sum() == 1)
            self.assertTrue(transitioned_2_v2.b.sum() == 4)
            transitioned_2_v3 = transitioned_2[
                transitioned_2.regeneration_delay == 3]
            self.assertTrue(transitioned_2_v3.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v3.reset_age.sum() == 3)
            self.assertTrue(transitioned_2_v3.a.sum() == 3)
            self.assertTrue(transitioned_2_v3.b.sum() == 5)
            transitioned_2_v4 = transitioned_2[
                transitioned_2.regeneration_delay == 0]
            self.assertTrue(transitioned_2_v4.area.sum() == 10.0 - 3.0)
            self.assertTrue(transitioned_2_v4.reset_age.sum() == -1)
            self.assertTrue(transitioned_2_v4.a.sum() == 1)
            self.assertTrue(transitioned_2_v4.b.sum() == 4)
