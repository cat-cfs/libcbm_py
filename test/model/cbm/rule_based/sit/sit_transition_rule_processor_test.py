import unittest
from mock import Mock
from unittest.mock import patch
from types import SimpleNamespace
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor"


class SITTransitionRuleProcessorTest(unittest.TestCase):

    def test_get_pre_dynamics_func(self):
        mock_cbm_vars = "mock_cbm_vars"
        mock_sit_transitions = "mock_sit_transitions"
        sit_transition_processor = Mock()
        process_transition_rules = Mock()

        sit_transition_processor.process_transition_rules = \
            process_transition_rules

        process_transition_rules.side_effect = \
            lambda sit_transitions, cbm_vars: cbm_vars
        func = sit_transition_rule_processor.get_pre_dynamics_func(
            sit_transition_processor, mock_sit_transitions
        )

        result = func(1, mock_cbm_vars)
        self.assertTrue(result == "mock_cbm_vars")
        process_transition_rules.assert_called_once_with(
            mock_sit_transitions, mock_cbm_vars)

    def test_state_variable_filter_func(self):

        with patch(PATCH_PATH + ".sit_stand_filter") as sit_stand_filter, \
             patch(PATCH_PATH + ".rule_filter") as rule_filter:

            create_state_filter_expression = Mock()
            create_state_filter_expression.side_effect = \
                lambda a, b: (
                    "mock_state_filter_expression",
                    "mock_state_filter_cols")
            sit_stand_filter.create_state_filter_expression = \
                create_state_filter_expression

            rule_filter.create_filter = Mock()
            rule_filter.create_filter.side_effect = \
                lambda **kwargs: "mock_filter_result"

            mock_tr_group_key = {}
            mock_state_variables = SimpleNamespace(age="mock_age")
            result = sit_transition_rule_processor.state_variable_filter_func(
                mock_tr_group_key, mock_state_variables)
            self.assertTrue(result == "mock_filter_result")
            create_state_filter_expression.assert_called_once_with(
                mock_tr_group_key, True)
            rule_filter.create_filter.assert_called_once_with(
                expression="mock_state_filter_expression",
                data={"age": "mock_age"},
                columns="mock_state_filter_cols"
            )
