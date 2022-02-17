import unittest
from mock import Mock
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor
from libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor import (
    SITTransitionRuleProcessor,
)

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor"


class SITTransitionRuleProcessorTest(unittest.TestCase):
    def test_state_variable_filter_func(self):

        with patch(
            PATCH_PATH + ".sit_stand_filter"
        ) as sit_stand_filter, patch(
            PATCH_PATH + ".rule_filter"
        ) as rule_filter:

            create_state_filter_expression = Mock()
            create_state_filter_expression.side_effect = (
                lambda a, b: "mock_state_filter_expression"
            )
            sit_stand_filter.create_state_filter_expression = (
                create_state_filter_expression
            )

            rule_filter.create_filter = Mock()
            rule_filter.create_filter.side_effect = (
                lambda **kwargs: "mock_filter_result"
            )

            mock_tr_group_key = {}
            mock_state_variables = SimpleNamespace(age="mock_age")
            result = sit_transition_rule_processor.state_variable_filter_func(
                mock_tr_group_key, mock_state_variables
            )
            self.assertTrue(result == "mock_filter_result")
            create_state_filter_expression.assert_called_once_with(
                mock_tr_group_key, True
            )
            rule_filter.create_filter.assert_called_once_with(
                expression="mock_state_filter_expression",
                data={"age": "mock_age"},
            )

    def test_sit_transition_rule_iterator(self):
        mock_sit_transitions = pd.DataFrame(
            {
                "c1": ["a1", "a1", "a2", "a2", "a3"],
                "c2": ["b1", "b1", "b2", "b2", "b3"],
                "min_age": [0, 0, 1, 1, 2],
                "max_age": [10, 10, 100, 100, 5],
                "disturbance_type_id": [1, 1, 2, 2, 3],
                "reset_age": [1, 2, 3, 4, 5],
                "percent": [50, 50, 30, 70, 100],
            }
        )
        classifier_names = ["c1", "c2"]
        result = sit_transition_rule_processor.sit_transition_rule_iterator(
            mock_sit_transitions, classifier_names
        )
        list_result = list(result)
        self.assertTrue(
            list_result[0][0]
            == {
                "c1": "a1",
                "c2": "b1",
                "min_age": 0,
                "max_age": 10,
                "disturbance_type_id": 1,
            }
        )
        self.assertTrue(
            list_result[0][1].equals(mock_sit_transitions.iloc[[0, 1]])
        )

        self.assertTrue(
            list_result[1][0]
            == {
                "c1": "a2",
                "c2": "b2",
                "min_age": 1,
                "max_age": 100,
                "disturbance_type_id": 2,
            }
        )
        self.assertTrue(
            list_result[1][1].equals(mock_sit_transitions.iloc[[2, 3]])
        )

        self.assertTrue(
            list_result[2][0]
            == {
                "c1": "a3",
                "c2": "b3",
                "min_age": 2,
                "max_age": 5,
                "disturbance_type_id": 3,
            }
        )
        self.assertTrue(
            list_result[2][1].equals(mock_sit_transitions.iloc[[4]])
        )

    def test_sit_transition_rule_iterator_error(self):
        """check that an error is raised if a grouped set of
        transition rules has a percent sum > 100
        """
        mock_sit_transitions = pd.DataFrame(
            {
                "c1": ["a1", "a1"],
                "c2": ["b1", "b1"],
                "min_age": [0, 0],
                "max_age": [10, 10],
                "disturbance_type_id": [1, 1],
                "reset_age": [1, 2],
                "percent": [51, 50],
            }
        )
        classifier_names = ["c1", "c2"]
        with self.assertRaises(ValueError):
            list(
                sit_transition_rule_processor.sit_transition_rule_iterator(
                    mock_sit_transitions, classifier_names
                )
            )

    def test_process_transition_rules(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=pd.DataFrame({"c1": [1], "c2": [1]}),
            parameters=pd.DataFrame({"reset_age": np.array([1, 2])}),
        )

        mock_transition_rule_processor = Mock()
        mock_apply_transition_rule = Mock()
        mock_transition_rule_processor.apply_transition_rule = (
            mock_apply_transition_rule
        )

        def test_apply_transition_rule(
            tr_group_key, tr_group, transition_mask, cbm_vars
        ):
            self.assertTrue(
                tr_group_key
                == {
                    "c1": "a1",
                    "c2": "b1",
                    "min_age": 0,
                    "max_age": 10,
                    "disturbance_type_id": 1,
                }
            )
            self.assertTrue(tr_group.equals(mock_sit_transitions))
            self.assertTrue(list(transition_mask) == [False])
            self.assertTrue(
                cbm_vars.classifiers.equals(mock_cbm_vars.classifiers)
            )

            return "mock_mask", "mock_cbm_vars_result"

        mock_apply_transition_rule.side_effect = test_apply_transition_rule
        s = SITTransitionRuleProcessor(mock_transition_rule_processor)
        mock_sit_transitions = pd.DataFrame(
            {
                "c1": ["a1", "a1"],
                "c2": ["b1", "b1"],
                "min_age": [0, 0],
                "max_age": [10, 10],
                "disturbance_type_id": [1, 1],
                "reset_age": [1, 2],
                "percent": [50, 50],
            }
        )

        cbm_vars_result = s.process_transition_rules(
            mock_sit_transitions, mock_cbm_vars
        )
        self.assertTrue(cbm_vars_result == "mock_cbm_vars_result")
        mock_apply_transition_rule.assert_called_once()
