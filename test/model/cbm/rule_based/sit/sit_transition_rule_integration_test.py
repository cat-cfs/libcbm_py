import unittest
from libcbm.input.sit import sit_cbm_factory
import test.model.cbm.rule_based.sit.sit_rule_based_integration_test_helpers as helpers  # noqa 501


class SITTransitionRuleIntegrationTest(unittest.TestCase):
    def test_single_stand_transition(self):
        sit_input = helpers.load_sit_input()
        sit_input.sit_data.transition_rules = helpers.initialize_transitions(
            sit_input,
            [
                {
                    "admin": "?",
                    "eco": "?",
                    "species": "sp",
                    "disturbance_type": "dist1",
                    "percent": 50,
                    "species_tr": "oak",
                    "regeneration_delay": 5,
                    "reset_age": 0,
                },
                {
                    "admin": "?",
                    "eco": "?",
                    "species": "sp",
                    "disturbance_type": "dist1",
                    "percent": 50,
                    "species_tr": "pn",
                    "regeneration_delay": 10,
                    "reset_age": -1,
                },
            ],
        )
        # change the post-dist1 species to 1/2 oak, 1/2 pine

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [{"admin": "a1", "eco": "e2", "species": "sp", "area": 5}],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # in order for the transition to occur, the disturbance type needs to
        # be set, normally this would be done beforehand by the sit rule based
        # events.
        cbm_vars.parameters["disturbance_type"].assign_all(
            helpers.get_disturbance_type_ids(
                sit.sit_data.disturbance_types, ["dist1"]
            )[0]
        )

        with helpers.get_rule_based_processor(sit) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.tr_func(
                cbm_vars=cbm_vars
            )

        self.assertTrue(cbm_vars_result.inventory.n_rows == 2)
