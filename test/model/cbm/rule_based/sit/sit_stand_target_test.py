import unittest
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
from libcbm.model.cbm.rule_based.sit import sit_stand_target


def call_test_function(
    mock_sit_event_row,
    mock_state_variables,
    mock_pools,
    mock_random_generator,
    mock_disturbance_production_func=None,
):

    mock_inventory = "inventory"
    mock_eligible = "eligible"
    create_target = sit_stand_target.create_sit_event_target_factory(
        sit_event_row=mock_sit_event_row,
        disturbance_production_func=mock_disturbance_production_func,
        random_generator=mock_random_generator,
    )
    mock_cbm_vars = SimpleNamespace(
        inventory=mock_inventory, pools=mock_pools, state=mock_state_variables
    )
    create_target(cbm_vars=mock_cbm_vars, eligible=mock_eligible)
    return mock_cbm_vars


PATCH_PREFIX = "libcbm.model.cbm.rule_based.sit.sit_stand_target"


class SITStandTargetTest(unittest.TestCase):
    """Tests for functions that compute CBM rule based disturbance event
    targets based on SIT disturbance event input.
    """

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_proportion_sort_proportion_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "PROPORTION_OF_EVERY_RECORD",
                "target_type": "Proportion",
                "target": 0.8,
                "disturbance_type": "fire",
            },
            mock_state_variables="mock_state_vars",
            mock_pools="mock_pools",
            mock_random_generator=None,
        )

        rule_target.proportion_sort_proportion_target.assert_called_once_with(
            proportion_target=0.8, inventory="inventory", eligible="eligible"
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_proportion_sort_area_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "PROPORTION_OF_EVERY_RECORD",
                "target_type": "Area",
                "target": 13,
                "disturbance_type": "fire",
            },
            mock_state_variables="mock_state_vars",
            mock_pools="mock_pools",
            mock_random_generator=None,
        )

        rule_target.proportion_area_target.assert_called_once_with(
            area_target_value=13,
            inventory="inventory",
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    @patch(f"{PATCH_PREFIX}.sit_rule_based_sort")
    def test_production_sort_area_target(
        self, sit_rule_based_sort, rule_target
    ):
        mock_sort_type = "mock_sort_type"
        mock_production = "mock_production"

        sit_rule_based_sort.is_production_sort.side_effect = lambda _: True
        sit_rule_based_sort.is_production_based.side_effect = lambda _: True
        sit_rule_based_sort.get_production_sort_value.side_effect = (
            lambda *args: "production_sort_value"
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 2)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        mock_sit_event_row = {
            "sort_type": mock_sort_type,
            "target_type": "Area",
            "target": 18,
            "disturbance_type_id": 2,
        }
        call_test_function(
            mock_sit_event_row=mock_sit_event_row,
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )
        sit_rule_based_sort.is_production_sort.assert_called_with(
            mock_sit_event_row
        )
        sit_rule_based_sort.is_production_based.assert_called_with(
            mock_sit_event_row
        )
        sit_rule_based_sort.get_production_sort_value.assert_called_with(
            mock_sort_type, mock_production, "pools"
        )
        rule_target.sorted_area_target.assert_called_once_with(
            area_target_value=18,
            sort_value="production_sort_value",
            inventory="inventory",
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    @patch(f"{PATCH_PREFIX}.sit_rule_based_sort")
    def test_non_production_area_targets(
        self, sit_rule_based_sort, rule_target
    ):
        sort_type = "mock_sort_type"
        mock_random_gen = "mock random generator"

        sit_rule_based_sort.is_production_sort.side_effect = lambda _: False
        sit_rule_based_sort.is_production_based.side_effect = lambda _: False
        sit_rule_based_sort.get_sort_value.side_effect = (
            lambda sort_type, cbm_vars, random_generator: "mock sort value"
        )

        mock_sit_event_row = {
            "sort_type": sort_type,
            "target_type": "Area",
            "target": 11,
            "disturbance_type": "fire",
        }
        mock_cbm_vars = call_test_function(
            mock_sit_event_row=mock_sit_event_row,
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=mock_random_gen,
        )

        sit_rule_based_sort.is_production_sort.assert_called_with(
            mock_sit_event_row
        )
        sit_rule_based_sort.is_production_based.assert_called_with(
            mock_sit_event_row
        )
        sit_rule_based_sort.get_sort_value.assert_called_with(
            sort_type, mock_cbm_vars, "mock random generator"
        )
        rule_target.sorted_area_target.assert_called_once_with(
            area_target_value=11,
            sort_value="mock sort value",
            inventory="inventory",
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    @patch(f"{PATCH_PREFIX}.sit_rule_based_sort")
    def test_proportion_sort_production_target(
        self, sit_rule_based_sort, rule_target
    ):
        mock_production = "mock_production"

        sit_rule_based_sort.is_production_sort.side_effect = lambda _: False
        sit_rule_based_sort.is_production_based.side_effect = lambda _: True
        sit_rule_based_sort.get_sort_value.side_effect = (
            lambda sort_type, cbm_vars, random_generator: "mock sort value"
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 90)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        mock_sit_event_row = {
            "sort_type": "PROPORTION_OF_EVERY_RECORD",
            "target_type": "Merchantable",
            "target": 17,
            "disturbance_type_id": 90,
            "efficiency": 55,
        }
        call_test_function(
            mock_sit_event_row=mock_sit_event_row,
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        sit_rule_based_sort.is_production_based.assert_called_with(
            mock_sit_event_row
        )

        rule_target.proportion_merch_target.assert_called_once_with(
            carbon_target=17,
            disturbance_production="mock_production",
            inventory="inventory",
            efficiency=55,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_merch_total_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1],
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 99)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        call_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_TOTAL",
                "target_type": "Merchantable",
                "target": 4,
                "disturbance_type_id": 99,
                "efficiency": 100,
            },
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )
        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=4,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_production.Total,
            efficiency=100,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_merch_sw_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1],
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 45)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = (
            mock_production.DisturbanceSoftProduction
            + mock_production.DisturbanceDOMProduction
        )

        call_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_SW",
                "target_type": "Merchantable",
                "target": 23,
                "disturbance_type_id": 45,
                "efficiency": 0.1,
            },
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=23,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=expected_sort_value,
            efficiency=0.1,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_merch_hw_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1],
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 73)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = (
            mock_production.DisturbanceHardProduction
            + mock_production.DisturbanceDOMProduction
        )

        call_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_HW",
                "target_type": "Merchantable",
                "target": 31,
                "disturbance_type_id": 73,
                "efficiency": 0.99,
            },
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=31,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=expected_sort_value,
            efficiency=0.99,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_random_sort_merch_target(self, rule_target):

        mock_pools = pd.DataFrame({"a": [12, 3, 4, 5]})

        def mock_random_gen(n_values):
            return [1] * n_values

        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1],
        )

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 43)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(list(cbm_vars.pools.a) == [12, 3, 4, 5])
            return mock_production

        call_test_function(
            mock_sit_event_row={
                "sort_type": "RANDOMSORT",
                "target_type": "Merchantable",
                "target": 31,
                "disturbance_type_id": 43,
                "efficiency": 0.99,
            },
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=mock_random_gen,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=31,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_random_gen(mock_pools.shape[0]),
            efficiency=0.99,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_total_stem_snag_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 15)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.SoftwoodStemSnag == [1, 2, 3])
            self.assertTrue(cbm_vars.pools.HardwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(
            SoftwoodStemSnag=[1, 2, 3], HardwoodStemSnag=[1, 2, 3]
        )

        call_test_function(
            mock_sit_event_row={
                "sort_type": "TOTALSTEMSNAG",
                "target_type": "Merchantable",
                "target": 37,
                "disturbance_type_id": 15,
                "efficiency": 1.0,
            },
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=37,
            disturbance_production=mock_production,
            inventory="inventory",
            # simply confirm the '+' operator is used on the correct pools
            sort_value=(
                mock_pools.SoftwoodStemSnag + mock_pools.HardwoodStemSnag
            ),
            efficiency=1.0,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_sw_stem_snag_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 57)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.SoftwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(SoftwoodStemSnag=[1, 2, 3])

        call_test_function(
            mock_sit_event_row={
                "sort_type": "SWSTEMSNAG",
                "target_type": "Merchantable",
                "target": 47,
                "disturbance_type_id": 57,
                "efficiency": 1.1,
            },
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=47,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_pools.SoftwoodStemSnag,
            efficiency=1.1,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_hw_stem_snag_sort_merch_target(self, rule_target):
        mock_production = SimpleNamespace(Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 9)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.HardwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(HardwoodStemSnag=[1, 2, 3])

        call_test_function(
            mock_sit_event_row={
                "sort_type": "HWSTEMSNAG",
                "target_type": "Merchantable",
                "target": 97,
                "disturbance_type_id": 9,
                "efficiency": 2.1,
            },
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func,
        )

        rule_target.sorted_merch_target.assert_called_once_with(
            carbon_target=97,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_pools.HardwoodStemSnag,
            efficiency=2.1,
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_age_sort_area_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "SORT_BY_HW_AGE",
                "target_type": "Area",
                "target": 100,
            },
            mock_state_variables=SimpleNamespace(age=[10, 2, 30]),
            mock_pools="pools",
            mock_random_generator=None,
        )

        rule_target.sorted_area_target.assert_called_once_with(
            area_target_value=100,
            sort_value=[10, 2, 30],
            inventory="inventory",
            eligible="eligible",
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_svoid_sort_proportion_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Proportion",
                "target": 100,
                "spatial_reference": 1000,
            },
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None,
        )

        rule_target.spatially_indexed_target.assert_called_once_with(
            identifier=1000, inventory="inventory"
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_svoid_sort_merch_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Merchantable",
                "target": 10,
                "spatial_reference": 4000,
            },
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None,
        )

        rule_target.spatially_indexed_target.assert_called_once_with(
            identifier=4000, inventory="inventory"
        )

    @patch(f"{PATCH_PREFIX}.rule_target")
    def test_svoid_sort_area_target(self, rule_target):
        call_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Area",
                "target": 130,
                "spatial_reference": 1050,
            },
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None,
        )

        rule_target.spatially_indexed_target.assert_called_once_with(
            identifier=1050, inventory="inventory"
        )
