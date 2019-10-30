import unittest
from mock import Mock
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor


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
