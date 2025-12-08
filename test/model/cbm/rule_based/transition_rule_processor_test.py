import unittest
from unittest.mock import patch
from mock import Mock
import pandas as pd
import numpy as np
from libcbm.storage import dataframe
from libcbm.storage import series
from types import SimpleNamespace
from libcbm.model.cbm.rule_based.transition_rule_processor import (
    TransitionRuleProcessor,
)

PATCH_PATH = "libcbm.model.cbm.rule_based.transition_rule_processor"


class TransitionRuleProcessorTest(unittest.TestCase):
    def test_single_record_transition(self):
        mock_classifier_config = {
            "classifiers": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"},
            ],
        }
        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_config,
            wildcard,
            transition_classifier_postfix,
        )

        tr_group = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "a": ["a1"],
                    "b": ["b1"],
                    "disturbance_type_id": [55],
                    "a_tr": ["a2"],
                    "b_tr": ["?"],
                    "regeneration_delay": [10],
                    "reset_age": [40],
                    "percent": [100],
                }
            )
        )
        transition_mask = series.from_numpy("", np.array([False], dtype=bool))
        mock_classifiers = dataframe.from_pandas(
            pd.DataFrame({"a": [1], "b": [3]})
        )
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [1.0],
                    "inventory_id": [20],
                    "parent_inventory_id": [8000],
                }
            )
        )
        mock_pools = dataframe.from_pandas(
            pd.DataFrame({"p0": [1], "p1": [4]})
        )
        mock_state_variables = dataframe.from_pandas(
            pd.DataFrame({"age": [0], "regeneration_delay": [0]})
        )
        mock_params = dataframe.from_pandas(
            pd.DataFrame({"disturbance_type": [0], "reset_age": [-1]})
        )
        mock_flux = dataframe.from_pandas(pd.DataFrame({"f1": [0], "f2": [0]}))
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            parameters=mock_params,
            flux=mock_flux,
        )

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filters.side_effect = (
                lambda *args: series.from_numpy(
                    "", np.array([True], dtype=bool)
                )
            )
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                tr_group, [], [1.0], transition_mask, mock_cbm_vars
            )
            self.assertTrue(transition_mask.to_list() == [True])
            self.assertTrue(
                cbm_vars.state["regeneration_delay"].to_list() == [10]
            )
            self.assertTrue(cbm_vars.state["age"].to_list() == [0])
            self.assertTrue(cbm_vars.parameters["reset_age"].to_list() == [40])

            # note the changed classifier id
            self.assertTrue(cbm_vars.classifiers["a"].to_list() == [2])

            # note unchanged since "?" was used for transition classifier value
            self.assertTrue(cbm_vars.classifiers["b"].to_list() == [3])

            self.assertTrue(cbm_vars.inventory["area"].to_list() == [1.0])

            self.assertTrue(
                cbm_vars.inventory["inventory_id"].to_list() == [20]
            )
            self.assertTrue(
                cbm_vars.inventory["parent_inventory_id"].to_list() == [8000]
            )
            self.assertTrue(cbm_vars.pools["p0"].to_list() == [1])
            self.assertTrue(cbm_vars.pools["p1"].to_list() == [4])
            self.assertTrue(cbm_vars.flux["f1"].to_list() == [0])
            self.assertTrue(cbm_vars.flux["f2"].to_list() == [0])

    def test_single_record_split_remainder_transition(self):
        mock_classifier_config = {
            "classifiers": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"},
            ],
        }
        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_config,
            wildcard,
            transition_classifier_postfix,
        )

        tr_group = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "a": ["a1"],
                    "b": ["?"],
                    "disturbance_type_id": [55],
                    "a_tr": ["a2"],
                    "b_tr": ["?"],
                    "regeneration_delay": [10],
                    "reset_age": [40],
                    "percent": [50],
                }
            )
        )
        transition_mask = series.from_numpy("", np.array([False], dtype=bool))
        mock_classifiers = dataframe.from_pandas(
            pd.DataFrame({"a": [1], "b": [3]})
        )
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [1.0],
                    "inventory_id": [7],
                    "parent_inventory_id": [2],
                }
            )
        )
        mock_pools = dataframe.from_pandas(
            pd.DataFrame({"p0": [1], "p1": [4]})
        )
        mock_state_variables = dataframe.from_pandas(
            pd.DataFrame({"age": [0], "regeneration_delay": [0]})
        )
        mock_params = dataframe.from_pandas(
            pd.DataFrame({"disturbance_type": [0], "reset_age": [-1]})
        )
        mock_flux = dataframe.from_pandas(pd.DataFrame({"f1": [0], "f2": [0]}))
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            parameters=mock_params,
            flux=mock_flux,
        )

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filters.side_effect = (
                lambda *args: series.from_numpy(
                    "", np.array([True], dtype=bool)
                )
            )
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                tr_group, [], [0.5, 0.5], transition_mask, mock_cbm_vars
            )

            self.assertTrue(transition_mask.to_list() == [True, True])
            self.assertTrue(
                cbm_vars.state["regeneration_delay"].to_list() == [10, 0]
            )
            self.assertTrue(cbm_vars.state["age"].to_list() == [0, 0])
            self.assertTrue(cbm_vars.parameters["reset_age"] == [40, -1])

            # note the changed classifier id for index 0, but not for index 1
            # (because of the remainder split)
            self.assertTrue(cbm_vars.classifiers["a"].to_list() == [2, 1])

            # note unchanged since "?" was used for transition classifier value
            self.assertTrue(cbm_vars.classifiers["b"].to_list() == [3, 3])

            self.assertTrue(cbm_vars.inventory["area"].to_list() == [0.5, 0.5])
            self.assertTrue(
                cbm_vars.inventory["inventory_id"].to_list() == [7, 8]
            )
            self.assertTrue(
                cbm_vars.inventory["parent_inventory_id"].to_list() == [2, 7]
            )

            # since pools are area densities the are just copied here
            self.assertTrue(cbm_vars.pools["p0"].to_list() == [1, 1])
            self.assertTrue(cbm_vars.pools["p1"].to_list() == [4, 4])

            self.assertTrue(cbm_vars.flux["f1"].to_list() == [0, 0])
            self.assertTrue(cbm_vars.flux["f2"].to_list() == [0, 0])

    def test_single_record_split_transition(self):
        mock_classifier_config = {
            "classifiers": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 2, "value": "b1"},
                {"id": 4, "classifier_id": 2, "value": "b2"},
            ],
        }

        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_config,
            wildcard,
            transition_classifier_postfix,
        )

        tr_group = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "a": ["a1", "a1"],
                    "b": ["?", "?"],
                    "disturbance_type_id": [55, 55],
                    "a_tr": ["a2", "a1"],
                    "b_tr": ["?", "b2"],
                    "regeneration_delay": [10, -1],
                    "reset_age": [40, 21],
                    "percent": [35, 65],
                }
            )
        )
        transition_mask = series.from_numpy("", np.array([False], dtype=bool))
        mock_classifiers = dataframe.from_pandas(
            pd.DataFrame({"a": [1], "b": [3]})
        )
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [1.0],
                    "inventory_id": [98000],
                    "parent_inventory_id": [1],
                }
            )
        )
        mock_pools = dataframe.from_pandas(
            pd.DataFrame({"p0": [33], "p1": [11]})
        )
        mock_state_variables = dataframe.from_pandas(
            pd.DataFrame({"age": [0], "regeneration_delay": [999]})
        )
        mock_params = dataframe.from_pandas(
            pd.DataFrame({"disturbance_type": [0], "reset_age": [-1]})
        )
        mock_flux = dataframe.from_pandas(
            pd.DataFrame({"f1": [10], "f2": [100]})
        )
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            parameters=mock_params,
            flux=mock_flux,
        )

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_rule_filter.evaluate_filters.side_effect = (
                lambda *args: series.from_numpy(
                    "", np.array([True], dtype=bool)
                )
            )
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                tr_group, [], [0.35, 0.65], transition_mask, mock_cbm_vars
            )

            self.assertTrue(transition_mask.to_list() == [True, True])
            self.assertTrue(
                cbm_vars.state["regeneration_delay"].to_list() == [10, 999]
            )
            self.assertTrue(cbm_vars.state["age"].to_list() == [0, 0])
            self.assertTrue(
                cbm_vars.parameters["reset_age"].to_list() == [40, 21]
            )
            self.assertTrue(cbm_vars.classifiers["a"].to_list() == [2, 1])
            self.assertTrue(cbm_vars.classifiers["b"].to_list() == [3, 4])
            self.assertTrue(
                cbm_vars.inventory["area"].to_list() == [0.35, 0.65]
            )
            self.assertTrue(
                cbm_vars.inventory["inventory_id"].to_list() == [98000, 98001]
            )
            self.assertTrue(
                cbm_vars.inventory["parent_inventory_id"].to_list()
                == [1, 98000]
            )
            self.assertTrue(cbm_vars.pools["p0"].to_list() == [33, 33])
            self.assertTrue(cbm_vars.pools["p1"].to_list() == [11, 11])
            self.assertTrue(cbm_vars.flux["f1"].to_list() == [10, 10])
            self.assertTrue(cbm_vars.flux["f2"].to_list() == [100, 100])

    def test_multiple_records_multiple_split_transitions(self):
        mock_classifier_config = {
            "classifiers": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "value": "a1"},
                {"id": 2, "classifier_id": 1, "value": "a2"},
                {"id": 3, "classifier_id": 1, "value": "a3"},
                {"id": 4, "classifier_id": 2, "value": "b1"},
                {"id": 5, "classifier_id": 2, "value": "b2"},
                {"id": 6, "classifier_id": 2, "value": "b3"},
            ],
        }

        wildcard = "?"
        transition_classifier_postfix = "_tr"
        tr_processor = TransitionRuleProcessor(
            mock_classifier_config,
            wildcard,
            transition_classifier_postfix,
        )

        tr_group = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "a": ["a1", "a1", "a1"],
                    "b": ["?", "?", "?"],
                    "disturbance_type_id": [55, 55, 55],
                    "a_tr": ["a2", "a1", "a3"],
                    "b_tr": ["?", "b1", "b2"],
                    "regeneration_delay": [1, 2, 3],
                    "reset_age": [1, 2, 3],
                    "percent": [10, 10, 10],
                }
            )
        )
        transition_mask = series.from_numpy("", np.array([False], dtype=bool))
        mock_classifiers = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "a": [
                        1,  # "a1" - eligible for transtion
                        2,  # "a2" - not eligible for transtion
                        1,  # "a1" - eligible for transtion
                    ],
                    "b": [3, 6, 4],  # all eligible (wildcard criteria)
                }
            )
        )
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "index": [0, 1, 2],
                    "area": [1.0, 5.0, 10.0],
                    "inventory_id": [3, 4, 5],
                    "parent_inventory_id": [-1, 1, 2],
                }
            )
        )
        mock_pools = dataframe.from_pandas(
            pd.DataFrame({"p0": [33, 22, 11], "p1": [11, 0, -11]})
        )
        mock_state_variables = dataframe.from_pandas(
            pd.DataFrame({"age": [0, 1, 2], "regeneration_delay": [0, 0, 0]})
        )
        mock_params = dataframe.from_pandas(
            pd.DataFrame(
                {"disturbance_type": [0, 0, 0], "reset_age": [-1, -1, -1]}
            )
        )
        mock_flux = dataframe.from_pandas(
            pd.DataFrame({"f1": [10, 20, 30], "f2": [100, 90, 80]})
        )
        mock_cbm_vars = SimpleNamespace(
            classifiers=mock_classifiers,
            inventory=mock_inventory,
            pools=mock_pools,
            state=mock_state_variables,
            parameters=mock_params,
            flux=mock_flux,
        )

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            # for the test, indexes 0 and 2 will be eligible
            mock_rule_filter.evaluate_filters.side_effect = (
                lambda *args: series.from_numpy(
                    "", np.array([True, False, True], dtype=bool)
                )
            )
            transition_mask, cbm_vars = tr_processor.apply_transition_rule(
                tr_group,
                [],
                [0.1, 0.1, 0.1, 0.7],
                transition_mask,
                mock_cbm_vars,
            )

            result = (
                cbm_vars.inventory.to_pandas()
                .join(cbm_vars.classifiers.to_pandas())
                .join(cbm_vars.state.to_pandas())
                .join(cbm_vars.parameters.to_pandas())
                .join(cbm_vars.pools.to_pandas())
                .join(cbm_vars.flux.to_pandas())
                .join(
                    pd.DataFrame(
                        {"transition_mask": transition_mask.to_numpy()}
                    )
                )
            )

            transitioned_1 = result[result["index"] == 0]
            non_transitioned = result[result["index"] == 1]
            transitioned_2 = result[result["index"] == 2]

            self.assertTrue(
                (
                    (transitioned_1.f1 == 10)
                    & (transitioned_1.f2 == 100)
                    & (transitioned_1.p0 == 33)
                    & (transitioned_1.p1 == 11)
                    & (transitioned_1.transition_mask.all())
                ).all()
            )
            self.assertTrue(
                (
                    (non_transitioned.f1 == 20)
                    & (non_transitioned.f2 == 90)
                    & (non_transitioned.p0 == 22)
                    & (non_transitioned.p1 == 0)
                    & (~non_transitioned.transition_mask.all())
                ).all()
            )
            self.assertTrue(
                (
                    (transitioned_2.f1 == 30)
                    & (transitioned_2.f2 == 80)
                    & (transitioned_2.p0 == 11)
                    & (transitioned_2.p1 == -11)
                    & (transitioned_2.transition_mask.all())
                ).all()
            )

            self.assertTrue(transitioned_1.shape[0] == 4)
            self.assertTrue(transitioned_1.area.sum() == 1.0)

            self.assertTrue(non_transitioned.shape[0] == 1)
            self.assertTrue(non_transitioned.area.sum() == 5.0)

            self.assertTrue(transitioned_2.shape[0] == 4)
            self.assertTrue(transitioned_2.area.sum() == 10.0)

            transitioned_1_v1 = transitioned_1[
                transitioned_1.regeneration_delay == 1
            ]
            self.assertTrue(transitioned_1_v1.area.sum() == 1.0 / 10.0)
            self.assertTrue(transitioned_1_v1.reset_age.sum() == 1)
            self.assertTrue(transitioned_1_v1.a.sum() == 2)
            self.assertTrue(transitioned_1_v1.b.sum() == 3)
            self.assertTrue(transitioned_1_v1.inventory_id.sum() == 3)
            self.assertTrue(transitioned_1_v1.parent_inventory_id.sum() == -1)
            transitioned_1_v2 = transitioned_1[
                transitioned_1.regeneration_delay == 2
            ]
            self.assertTrue(transitioned_1_v2.area.sum() == 1.0 / 10.0)
            self.assertTrue(transitioned_1_v2.reset_age.sum() == 2)
            self.assertTrue(transitioned_1_v2.a.sum() == 1)
            self.assertTrue(transitioned_1_v2.b.sum() == 4)
            self.assertTrue(transitioned_1_v2.inventory_id.sum() == 6)
            self.assertTrue(transitioned_1_v2.parent_inventory_id.sum() == 3)
            transitioned_1_v3 = transitioned_1[
                transitioned_1.regeneration_delay == 3
            ]
            self.assertTrue(transitioned_1_v3.area.sum() == 1.0 / 10.0)
            self.assertTrue(transitioned_1_v3.reset_age.sum() == 3)
            self.assertTrue(transitioned_1_v3.a.sum() == 3)
            self.assertTrue(transitioned_1_v3.b.sum() == 5)
            self.assertTrue(transitioned_1_v3.inventory_id.sum() == 8)
            self.assertTrue(transitioned_1_v3.parent_inventory_id.sum() == 3)

            transitioned_1_v4 = transitioned_1[
                transitioned_1.regeneration_delay == 0
            ]
            self.assertTrue(transitioned_1_v4.area.sum() == 1.0 - (3.0 / 10.0))
            self.assertTrue(transitioned_1_v4.reset_age.sum() == -1)
            self.assertTrue(transitioned_1_v4.a.sum() == 1)
            self.assertTrue(transitioned_1_v4.b.sum() == 3)
            self.assertTrue(transitioned_1_v4.inventory_id.sum() == 10)
            self.assertTrue(transitioned_1_v4.parent_inventory_id.sum() == 3)

            transitioned_2_v1 = transitioned_2[
                transitioned_2.regeneration_delay == 1
            ]
            self.assertTrue(transitioned_2_v1.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v1.reset_age.sum() == 1)
            self.assertTrue(transitioned_2_v1.a.sum() == 2)
            self.assertTrue(transitioned_2_v1.b.sum() == 4)
            self.assertTrue(transitioned_2_v1.inventory_id.sum() == 5)
            self.assertTrue(transitioned_2_v1.parent_inventory_id.sum() == 2)
            transitioned_2_v2 = transitioned_2[
                transitioned_2.regeneration_delay == 2
            ]
            self.assertTrue(transitioned_2_v2.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v2.reset_age.sum() == 2)
            self.assertTrue(transitioned_2_v2.a.sum() == 1)
            self.assertTrue(transitioned_2_v2.b.sum() == 4)
            self.assertTrue(transitioned_2_v2.inventory_id.sum() == 7)
            self.assertTrue(transitioned_2_v2.parent_inventory_id.sum() == 5)
            transitioned_2_v3 = transitioned_2[
                transitioned_2.regeneration_delay == 3
            ]
            self.assertTrue(transitioned_2_v3.area.sum() == 1.0)
            self.assertTrue(transitioned_2_v3.reset_age.sum() == 3)
            self.assertTrue(transitioned_2_v3.a.sum() == 3)
            self.assertTrue(transitioned_2_v3.b.sum() == 5)
            self.assertTrue(transitioned_2_v3.inventory_id.sum() == 9)
            self.assertTrue(transitioned_2_v3.parent_inventory_id.sum() == 5)
            transitioned_2_v4 = transitioned_2[
                transitioned_2.regeneration_delay == 0
            ]
            self.assertTrue(transitioned_2_v4.area.sum() == 10.0 - 3.0)
            self.assertTrue(transitioned_2_v4.reset_age.sum() == -1)
            self.assertTrue(transitioned_2_v4.a.sum() == 1)
            self.assertTrue(transitioned_2_v4.b.sum() == 4)
            self.assertTrue(transitioned_2_v4.inventory_id.sum() == 11)
            self.assertTrue(transitioned_2_v4.parent_inventory_id.sum() == 5)
