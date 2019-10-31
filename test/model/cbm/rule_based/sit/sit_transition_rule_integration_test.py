import unittest
import numpy as np

import test.model.cbm.rule_based.sit.sit_rule_based_integration_test_helpers \
    as helpers


class SITTransitionRuleIntegrationTest(unittest.TestCase):

    def test_single_stand_transition(self):
        sit = helpers.load_sit_data()
        sit.sit_data.transition_rules = helpers.initialize_transitions(sit, [
            {"admin": "?", "eco": "?", "species": "sp",
             "disturbance_type": "fire", "percent": 50, "species_tr": "oak",
             "regeneration_delay": 5, "reset_age": 0},
            {"admin": "?", "eco": "?", "species": "sp",
             "disturbance_type": "fire", "percent": 50, "species_tr": "pn",
             "regeneration_delay": 10, "reset_age": -1}
            ])
        # change the post-fire species to 1/2 oak, 1/2 pine

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # in order for the transition to occur, the disturbance type needs to
        # be set, normally this would be done beforehand by the sit rule based
        # events.
        cbm_vars.params.disturbance_type = helpers.FIRE_ID

        pre_dynamics_func = helpers.get_transition_rules_pre_dynamics_func(
            sit)
        cbm_vars_result = pre_dynamics_func(0, cbm_vars=cbm_vars)

        self.assertTrue(cbm_vars_result.inventory.shape[0] == 2)
