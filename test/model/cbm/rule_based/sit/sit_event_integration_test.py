import unittest
from mock import Mock
import numpy as np

import test.model.cbm.rule_based.sit.sit_rule_based_integration_test_helpers \
    as helpers

FIRE_ID = 1
CLEARCUT_ID = 3
DEFORESTATION_ID = 7


class SITEventIntegrationTest(unittest.TestCase):

    def test_rule_based_area_target_age_sort(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
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

    def test_rule_based_area_target_age_sort_unrealized(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        mock_on_unrealized = Mock()

        sit = helpers.load_sit_data()
        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a2", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1}
        ])

        # record at index 1 is the only eligible record meaning the above event
        # will be unrealized with a shortfall of 5
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

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        # records 0 and 3 are the disturbed records: both are eligible, they
        # are the oldest stands, and together they exactly satisfy the target.
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) == [0, 1, 0, 0])

        mock_args, _ = mock_on_unrealized.call_args
        self.assertTrue(mock_args[0] == 5)
        expected = sit.sit_data.disturbance_events.to_dict("records")[0]
        expected["disturbance_type_id"] = 1
        diff = set(mock_args[1].items()) ^ set(expected.items())
        self.assertTrue(len(diff) == 0)

    def test_rule_based_area_target_age_sort_multiple_event(self):
        """Check interactions between two age sort/area target events
        """
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()
        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1},
            {"admin": "?", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1},
        ])
        # the second of the above events will match all records, and it will
        # occur first since fire happens before clearcut

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e3", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state.age = np.array([100, 99, 98, 97, 96])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, FIRE_ID, 0, CLEARCUT_ID, CLEARCUT_ID])

    def test_rule_based_area_target_age_sort_split(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()
        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 6, "disturbance_type": "fire", "time_step": 1}
        ])
        # since the target is 6, one of the 2 inventory records below needs to
        # be split
        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # since the sort is by age, the first record will be fully disturbed
        # and the second will be split into 1 and 4 hectare stands.
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, FIRE_ID, 0])

        self.assertTrue(cbm_vars.pools.shape[0] == 3)
        self.assertTrue(cbm_vars.flux_indicators.shape[0] == 3)
        self.assertTrue(cbm_vars.state.shape[0] == 3)
        # note the age sort order caused the first record to split
        self.assertTrue(list(cbm_vars.inventory.area) == [1, 5, 4])

    def test_rule_based_merch_target_age_sort(self):
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()

        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1}
        ])

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # 1 tonnes C/ha * 10 ha total = 10 tonnes C
        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized, helpers.get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID])

    def test_rule_based_merch_target_age_sort_unrealized(self):
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()

        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1}
        ])

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 3},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 4},
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 2},
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        # 1 tonnes C/ha * (3+4+2) ha total = 9 tonnes C available for event,
        # with target = 10, therefore the expected shortfall is 1
        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100, 98])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized, helpers.get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID, CLEARCUT_ID])

        mock_args, _ = mock_on_unrealized.call_args
        # confirm expected shortfall (in tonnes C)
        self.assertTrue(mock_args[0] == 1)
        expected = sit.sit_data.disturbance_events.to_dict("records")[0]
        expected["disturbance_type_id"] = 3
        diff = set(mock_args[1].items()) ^ set(expected.items())
        self.assertTrue(len(diff) == 0)

    def test_rule_based_merch_target_age_sort_split(self):
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()

        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 7, "disturbance_type": "clearcut",
             "time_step": 4}
        ])

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            # the remaining target after 7 - 5 = 2, so 2/5ths of the area
            # of this stand will be disturbed
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            # this entire record will be disturbed first (see age sort)
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized, helpers.get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=4, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID, 0])
        self.assertTrue(cbm_vars.pools.shape[0] == 3)
        self.assertTrue(cbm_vars.flux_indicators.shape[0] == 3)
        self.assertTrue(cbm_vars.state.shape[0] == 3)
        # note the age sort order caused the first record to split
        self.assertTrue(list(cbm_vars.inventory.area) == [2, 5, 3])

    def test_rule_based_multiple_target_types(self):
        mock_on_unrealized = Mock()
        sit = helpers.load_sit_data()

        sit.sit_data.disturbance_events = helpers.initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "MERCHCSORT_TOTAL", "target_type": "Merchantable",
             "target": 100, "disturbance_type": "clearcut",
             "time_step": 100},
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 20, "disturbance_type": "deforestation",
             "time_step": 100},
            # this event will occur first
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "RANDOMSORT", "target_type": "Area",
             "target": 20, "disturbance_type": "fire",
             "time_step": 100},
        ])

        sit.sit_data.inventory = helpers.initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 1000},
        ])

        cbm_vars = helpers.setup_cbm_vars(sit)

        cbm_vars.pools.HardwoodMerch = 1.0
        cbm_vars.state.age = np.array([50])

        pre_dynamics_func = helpers.get_events_pre_dynamics_func(
            sit, mock_on_unrealized, helpers.get_parameters_factory(),
            random_func=np.ones)
        cbm_vars_result = pre_dynamics_func(time_step=100, cbm_vars=cbm_vars)
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, CLEARCUT_ID, DEFORESTATION_ID, 0])
        self.assertTrue(list(cbm_vars.inventory.area) == [20, 100, 20, 860])
