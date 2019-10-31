import unittest
from mock import Mock
import numpy as np

import test.model.cbm.rule_based.sit.sit_rule_based_integration_test_helpers \
    as helpers


class SITTransitionRuleIntegrationTest(unittest.TestCase):

    def test_single_stand_transition(self):
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()
        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1}
        ])

        # records 0, 2, and 3 match, and 1 does not.  The target is 10, so
        # 2 of the 3 eligible records will be disturbed
        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state.age = np.array([99, 100, 98, 100])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        # records 0 and 3 are the disturbed records: both are eligible, they
        # are the oldest stands, and together they exactly satisfy the target.
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
                [FIRE_ID, 0, 0, FIRE_ID])