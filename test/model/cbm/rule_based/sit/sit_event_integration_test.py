import unittest
import numpy as np
from libcbm.input.sit import sit_cbm_factory
from libcbm.storage import series
import test.model.cbm.rule_based.sit.sit_rule_based_integration_test_helpers as helpers  # noqa 501


class SITEventIntegrationTest(unittest.TestCase):
    def test_rule_based_area_target_age_sort(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        sit_input = helpers.load_sit_input()
        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 10,
                    "disturbance_type": "dist1",
                    "time_step": 1,
                }
            ],
        )

        # records 0, 2, and 3 match, and 1 does not.  The target is 10, so
        # 2 of the 3 eligible records will be disturbed
        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            ],
        )

        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state["age"].assign_all(np.array([99, 100, 98, 100]))

        with helpers.get_rule_based_processor(sit) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            # records 0 and 3 are the disturbed records: both are eligible,
            # they are the oldest stands, and together they exactly satisfy
            # the target.
            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types,
                    ["dist1", None, None, "dist1"],
                )
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                1
            ].iloc[0]
        self.assertTrue(stats_row["total_eligible_value"] == 15.0)
        self.assertTrue(stats_row["total_achieved"] == 10.0)
        self.assertTrue(stats_row["shortfall"] == 0.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 2)
        self.assertTrue(stats_row["num_splits"] == 0)
        self.assertTrue(stats_row["num_eligible"] == 3)

    def test_rule_based_area_target_age_sort_unrealized(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """

        sit_input = helpers.load_sit_input()
        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a2",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 10,
                    "disturbance_type": "dist1",
                    "time_step": 1,
                }
            ],
        )

        # record at index 1 is the only eligible record meaning the above event
        # will be unrealized with a shortfall of 5
        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state["age"].assign_all(np.array([99, 100, 98, 100]))

        with helpers.get_rule_based_processor(sit) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            # records 0 and 3 are the disturbed records: both are eligible,
            # they are the oldest stands, and together they exactly satisfy
            # the target.
            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types, [None, "dist1", None, None]
                )
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                1
            ].iloc[0]

        self.assertTrue(stats_row["total_eligible_value"] == 5.0)
        self.assertTrue(stats_row["total_achieved"] == 5.0)
        self.assertTrue(stats_row["shortfall"] == 5.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 1)
        self.assertTrue(stats_row["num_splits"] == 0)
        self.assertTrue(stats_row["num_eligible"] == 1)

    def test_rule_based_area_target_age_sort_multiple_event(self):
        """Check interactions between two age sort/area target events"""

        sit_input = helpers.load_sit_input()
        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 10,
                    "disturbance_type": "dist2",
                    "time_step": 1,
                },
                {
                    "admin": "?",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 10,
                    "disturbance_type": "dist1",
                    "time_step": 1,
                },
            ],
        )
        # the second of the above events will match all records, and it will
        # occur first since fire happens before clearcut

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e3", "species": "sp", "area": 5},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state["age"].assign_all(np.array([100, 99, 98, 97, 96]))

        with helpers.get_rule_based_processor(sit) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            expected_disturbance_types = helpers.get_disturbance_type_ids(
                sit.sit_data.disturbance_types,
                ["dist1", "dist1", None, "dist2", "dist2"],
            )

            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == expected_disturbance_types
            )

            stats = sit_rule_based_processor.sit_event_stats_by_timestep[1]
        self.assertTrue(stats.iloc[0]["sit_event_index"] == 1)
        self.assertTrue(stats.iloc[0]["total_eligible_value"] == 25.0)
        self.assertTrue(stats.iloc[0]["total_achieved"] == 10.0)
        self.assertTrue(stats.iloc[0]["shortfall"] == 0.0)
        self.assertTrue(stats.iloc[0]["num_records_disturbed"] == 2)
        self.assertTrue(stats.iloc[0]["num_splits"] == 0)
        self.assertTrue(stats.iloc[0]["num_eligible"] == 5)

        self.assertTrue(stats.iloc[1]["sit_event_index"] == 0)
        # less area is available as a result of the first event
        self.assertTrue(stats.iloc[1]["total_eligible_value"] == 10.0)
        self.assertTrue(stats.iloc[1]["total_achieved"] == 10.0)
        self.assertTrue(stats.iloc[1]["shortfall"] == 0.0)
        self.assertTrue(stats.iloc[1]["num_records_disturbed"] == 2)
        self.assertTrue(stats.iloc[1]["num_splits"] == 0)
        self.assertTrue(stats.iloc[1]["num_eligible"] == 2)

    def test_rule_based_area_target_age_sort_split(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """

        sit_input = helpers.load_sit_input()
        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 6,
                    "disturbance_type": "dist1",
                    "time_step": 1,
                }
            ],
        )
        # since the target is 6, one of the 2 inventory records below needs to
        # be split
        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # since the sort is by age, the first record will be fully disturbed
        # and the second will be split into 1 and 4 hectare stands.
        cbm_vars.state["age"].assign_all(np.array([99, 100]))

        with helpers.get_rule_based_processor(sit) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types, ["dist1", "dist1", None]
                )
            )

            self.assertTrue(cbm_vars_result.pools.n_rows == 3)
            self.assertTrue(cbm_vars_result.flux.n_rows == 3)
            self.assertTrue(cbm_vars_result.state.n_rows == 3)
            # note the age sort order caused the first record to split
            self.assertTrue(
                cbm_vars_result.inventory["area"].to_list() == [1, 5, 4]
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                1
            ].iloc[0]
        self.assertTrue(stats_row["total_eligible_value"] == 10.0)
        self.assertTrue(stats_row["total_achieved"] == 6.0)
        self.assertTrue(stats_row["shortfall"] == 0.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 2)
        self.assertTrue(stats_row["num_splits"] == 1)
        self.assertTrue(stats_row["num_eligible"] == 2)

    def test_rule_based_merch_target_age_sort(self):

        sit_input = helpers.load_sit_input()

        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_HW_AGE",
                    "target_type": "Merchantable",
                    "target": 10,
                    "disturbance_type": "dist2",
                    "time_step": 1,
                }
            ],
        )

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # 1 tonnes C/ha * 10 ha total = 10 tonnes C
        cbm_vars.pools["SoftwoodMerch"].assign_all(1.0)
        cbm_vars.state["age"].assign_all(np.array([99, 100]))

        with helpers.get_rule_based_processor(
            sit,
            random_func=None,
            parameters_factory=helpers.get_parameters_factory(sit),
        ) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                1
            ].iloc[0]
        self.assertTrue(stats_row["total_eligible_value"] == 10.0)
        self.assertTrue(stats_row["total_achieved"] == 10.0)
        self.assertTrue(stats_row["shortfall"] == 0.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 2)
        self.assertTrue(stats_row["num_splits"] == 0)
        self.assertTrue(stats_row["num_eligible"] == 2)

        self.assertTrue(
            cbm_vars_result.parameters["disturbance_type"].to_list()
            == helpers.get_disturbance_type_ids(
                sit.sit_data.disturbance_types, ["dist2", "dist2"]
            )
        )

    def test_rule_based_merch_target_age_sort_unrealized(self):
        sit_input = helpers.load_sit_input()

        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_HW_AGE",
                    "target_type": "Merchantable",
                    "target": 10,
                    "disturbance_type": "dist2",
                    "time_step": 1,
                }
            ],
        )

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 3},
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 4},
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 2},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        # 1 tonnes C/ha * (3+4+2) ha total = 9 tonnes C available for event,
        # with target = 10, therefore the expected shortfall is 1
        cbm_vars.pools["SoftwoodMerch"].assign_all(1.0)
        cbm_vars.state["age"].assign_all(np.array([99, 100, 98]))

        with helpers.get_rule_based_processor(
            sit,
            random_func=None,
            parameters_factory=helpers.get_parameters_factory(sit),
        ) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=1, cbm_vars=cbm_vars
            )

            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types, ["dist2", "dist2", "dist2"]
                )
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                1
            ].iloc[0]
        self.assertTrue(stats_row["total_eligible_value"] == 9.0)
        self.assertTrue(stats_row["total_achieved"] == 9.0)
        self.assertTrue(stats_row["shortfall"] == 1.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 3)
        self.assertTrue(stats_row["num_splits"] == 0)
        self.assertTrue(stats_row["num_eligible"] == 3)

    def test_rule_based_merch_target_age_sort_split(self):
        sit_input = helpers.load_sit_input()

        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_HW_AGE",
                    "target_type": "Merchantable",
                    "target": 7,
                    "disturbance_type": "dist2",
                    "time_step": 4,
                }
            ],
        )

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                # the remaining target after 7 - 5 = 2, so 2/5ths of the area
                # of this stand will be disturbed
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
                # this entire record will be disturbed first (see age sort)
                {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            ],
        )
        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        cbm_vars.pools["SoftwoodMerch"].assign_all(1.0)
        cbm_vars.state["age"].assign_all(np.array([99, 100]))

        with helpers.get_rule_based_processor(
            sit,
            random_func=None,
            parameters_factory=helpers.get_parameters_factory(sit),
        ) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=4, cbm_vars=cbm_vars
            )

            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types, ["dist2", "dist2", None]
                )
            )

            self.assertTrue(cbm_vars_result.pools.n_rows == 3)
            self.assertTrue(cbm_vars_result.flux.n_rows == 3)
            self.assertTrue(cbm_vars_result.state.n_rows == 3)
            # note the age sort order caused the first record to split
            self.assertTrue(
                cbm_vars_result.inventory["area"].to_list() == [2, 5, 3]
            )

            stats_row = sit_rule_based_processor.sit_event_stats_by_timestep[
                4
            ].iloc[0]
        self.assertTrue(stats_row["total_eligible_value"] == 10.0)
        self.assertTrue(stats_row["total_achieved"] == 7.0)
        self.assertTrue(stats_row["shortfall"] == 0.0)
        self.assertTrue(stats_row["num_records_disturbed"] == 2)
        self.assertTrue(stats_row["num_splits"] == 1)
        self.assertTrue(stats_row["num_eligible"] == 2)

    def test_rule_based_multiple_target_types(self):
        sit_input = helpers.load_sit_input()

        sit_input.sit_data.disturbance_events = helpers.initialize_events(
            sit_input,
            [
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "MERCHCSORT_TOTAL",
                    "target_type": "Merchantable",
                    "target": 100,
                    "disturbance_type": "dist2",
                    "time_step": 100,
                },
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "SORT_BY_SW_AGE",
                    "target_type": "Area",
                    "target": 20,
                    "disturbance_type": "dist3",
                    "time_step": 100,
                },
                # this event will occur first
                {
                    "admin": "a1",
                    "eco": "?",
                    "species": "sp",
                    "sort_type": "RANDOMSORT",
                    "target_type": "Area",
                    "target": 20,
                    "disturbance_type": "dist1",
                    "time_step": 100,
                },
            ],
        )

        sit_input.sit_data.inventory = helpers.initialize_inventory(
            sit_input,
            [
                {"admin": "a1", "eco": "e1", "species": "sp", "area": 1000},
            ],
        )

        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        cbm_vars = helpers.setup_cbm_vars(sit)

        cbm_vars.pools["HardwoodMerch"].assign_all(1.0)
        cbm_vars.state["age"].assign_all(np.array([50]))

        with helpers.get_rule_based_processor(
            sit,
            random_func=lambda size: series.from_numpy(None, np.ones(size)),
            parameters_factory=helpers.get_parameters_factory(sit),
        ) as sit_rule_based_processor:
            cbm_vars_result = sit_rule_based_processor.dist_func(
                time_step=100, cbm_vars=cbm_vars
            )
            self.assertTrue(
                cbm_vars_result.parameters["disturbance_type"].to_list()
                == helpers.get_disturbance_type_ids(
                    sit.sit_data.disturbance_types,
                    ["dist1", "dist2", "dist3", None],
                )
            )

            self.assertTrue(
                cbm_vars_result.inventory["area"].to_list()
                == [20, 100, 20, 860]
            )

            stats = sit_rule_based_processor.sit_event_stats_by_timestep[100]
        self.assertTrue(stats.iloc[0]["total_eligible_value"] == 1000.0)
        self.assertTrue(stats.iloc[0]["total_achieved"] == 20.0)
        self.assertTrue(stats.iloc[0]["shortfall"] == 0.0)
        self.assertTrue(stats.iloc[0]["num_records_disturbed"] == 1)
        self.assertTrue(stats.iloc[0]["num_splits"] == 1)
        self.assertTrue(stats.iloc[0]["num_eligible"] == 1)

        self.assertTrue(stats.iloc[1]["total_eligible_value"] == 980.0)
        self.assertTrue(stats.iloc[1]["total_achieved"] == 100.0)
        self.assertTrue(stats.iloc[1]["shortfall"] == 0.0)
        self.assertTrue(stats.iloc[1]["num_records_disturbed"] == 1)
        self.assertTrue(stats.iloc[1]["num_splits"] == 1)
        self.assertTrue(stats.iloc[1]["num_eligible"] == 1)

        self.assertTrue(stats.iloc[2]["total_eligible_value"] == 880.0)
        self.assertTrue(stats.iloc[2]["total_achieved"] == 20.0)
        self.assertTrue(stats.iloc[2]["shortfall"] == 0.0)
        self.assertTrue(stats.iloc[2]["num_records_disturbed"] == 1)
        self.assertTrue(stats.iloc[2]["num_splits"] == 1)
        self.assertTrue(stats.iloc[2]["num_eligible"] == 1)
