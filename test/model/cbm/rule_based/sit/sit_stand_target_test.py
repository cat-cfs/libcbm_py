import unittest
from types import SimpleNamespace
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based.sit import sit_stand_target
from libcbm.model.cbm.rule_based import rule_target


def get_test_function(mock_sit_event_row, mock_state_variables, mock_pools,
                      mock_random_generator,
                      mock_disturbance_production_func=None):
    mock_rule_target = Mock(spec=rule_target)

    mock_inventory = "inventory"
    mock_eligible = "eligible"
    create_target = sit_stand_target.create_sit_event_target_factory(
        rule_target=mock_rule_target,
        sit_event_row=mock_sit_event_row,
        disturbance_production_func=mock_disturbance_production_func,
        random_generator=mock_random_generator
        )
    mock_cbm_vars = SimpleNamespace(
        inventory=mock_inventory,
        pools=mock_pools,
        state=mock_state_variables)
    create_target(
        cbm_vars=mock_cbm_vars,
        eligible=mock_eligible)

    return mock_rule_target


class SITStandTargetTest(unittest.TestCase):
    """Tests for functions that compute CBM rule based disturbance event
    targets based on SIT disturbance event input.
    """

    def test_create_sit_event_target_proportion_sort_area_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "PROPORTION_OF_EVERY_RECORD",
                "target_type": "Area",
                "target": 13,
                "disturbance_type": "fire"},
            mock_state_variables="mock_state_vars",
            mock_pools="mock_pools",
            mock_random_generator=None
        ).proportion_area_target.assert_called_once_with(
            area_target_value=13,
            inventory="inventory",
            eligible="eligible",
        )

    def test_create_sit_event_target_merch_total_sort_area_target(self):

        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 2)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_TOTAL",
                "target_type": "Area",
                "target": 18,
                "disturbance_type_id": 2},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_area_target.assert_called_once_with(
            area_target_value=18,
            sort_value=mock_production.Total,
            inventory="inventory",
            eligible="eligible",
        )

    def test_create_sit_event_target_merch_sw_sort_area_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 4000)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = \
            mock_production.DisturbanceSoftProduction + \
            mock_production.DisturbanceDOMProduction

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_SW",
                "target_type": "Area",
                "target": 18,
                "disturbance_type_id": 4000},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_area_target.assert_called_once_with(
            area_target_value=18,
            sort_value=expected_sort_value,
            inventory="inventory",
            eligible="eligible",
        )

    def test_create_sit_event_target_merch_hw_sort_area_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 100)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = \
            mock_production.DisturbanceHardProduction + \
            mock_production.DisturbanceDOMProduction

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_HW",
                "target_type": "Area",
                "target": 19,
                "disturbance_type_id": 100},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_area_target.assert_called_once_with(
            area_target_value=19,
            sort_value=expected_sort_value,
            inventory="inventory",
            eligible="eligible")

    def test_create_sit_event_target_random_sort_area_target(self):
        mock_pools = pd.DataFrame({"a": [12, 3, 4, 5]})

        def mock_random_gen(n_values):
            return [1] * n_values

        get_test_function(
            mock_sit_event_row={
                "sort_type": "RANDOMSORT",
                "target_type": "Area",
                "target": 11,
                "disturbance_type": "fire"},
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=mock_random_gen
        ).sorted_area_target.assert_called_once_with(
            area_target_value=11,
            sort_value=mock_random_gen(mock_pools.shape[0]),
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_total_stem_snag_sort_area_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "TOTALSTEMSNAG",
                "target_type": "Area",
                "target": 50,
                "disturbance_type": "fire"},
            mock_state_variables="mock_state_vars",
            mock_pools=SimpleNamespace(
                SoftwoodStemSnag=[1, 2, 3, 4],
                HardwoodStemSnag=[5, 6, 7, 8]),
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=50,
            # since it's difficult for mock to test with
            # pd.DataSeries (simple equality won't work)
            # just check that the '+' operater was used.
            sort_value=[1, 2, 3, 4] + [5, 6, 7, 8],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_sw_stem_snag_sort_area_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SWSTEMSNAG",
                "target_type": "Area",
                "target": 1,
                "disturbance_type": "fire"},
            mock_state_variables="mock_state_vars",
            mock_pools=SimpleNamespace(SoftwoodStemSnag=[1, 2, 3, 4]),
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=1,
            sort_value=[1, 2, 3, 4],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_hw_stem_snag_sort_area_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "HWSTEMSNAG",
                "target_type": "Area",
                "target": 1,
                "disturbance_type": "fire"},
            mock_state_variables="mock_state_vars",
            mock_pools=SimpleNamespace(HardwoodStemSnag=[1, 2, 3, 4]),
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=1,
            sort_value=[1, 2, 3, 4],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_swage_sort_area_target(self):
        """confirm state_variable.age is used as a sort value
        """
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SORT_BY_SW_AGE",
                "target_type": "Area",
                "target": 100,
                "disturbance_type": "fire"},
            mock_state_variables=SimpleNamespace(age=[10, 2, 30]),
            mock_pools="pools",
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=100,
            sort_value=[10, 2, 30],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_hwage_sort_area_target(self):
        """confirm state_variable.age is used as a sort value
        """
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SORT_BY_HW_AGE",
                "target_type": "Area",
                "target": 100,
                "disturbance_type": "fire"},
            mock_state_variables=SimpleNamespace(age=[10, 2, 30]),
            mock_pools="pools",
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=100,
            sort_value=[10, 2, 30],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_proportion_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 90)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        get_test_function(
            mock_sit_event_row={
                "sort_type": "PROPORTION_OF_EVERY_RECORD",
                "target_type": "Merchantable",
                "target": 17,
                "disturbance_type_id": 90,
                "efficiency": 55},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).proportion_merch_target.assert_called_once_with(
            carbon_target=17,
            disturbance_production=mock_production.Total,
            inventory="inventory",
            efficiency=55,
            eligible="eligible"
        )

    def test_create_sit_event_target_merch_total_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 99)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_TOTAL",
                "target_type": "Merchantable",
                "target": 4,
                "disturbance_type_id": 99,
                "efficiency": 100},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=4,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_production.Total,
            efficiency=100,
            eligible="eligible"
        )

    def test_create_sit_event_target_merch_sw_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars,  disturbance_type_id):
            self.assertTrue(disturbance_type_id == 45)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = \
            mock_production.DisturbanceSoftProduction + \
            mock_production.DisturbanceDOMProduction

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_SW",
                "target_type": "Merchantable",
                "target": 23,
                "disturbance_type_id": 45,
                "efficiency": 0.1},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=23,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=expected_sort_value,
            efficiency=0.1,
            eligible="eligible"
        )

    def test_create_sit_event_target_merch_hw_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 73)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools == "pools")
            return mock_production

        # tests that the + operator is used for the correct production fields
        expected_sort_value = \
            mock_production.DisturbanceHardProduction + \
            mock_production.DisturbanceDOMProduction

        get_test_function(
            mock_sit_event_row={
                "sort_type": "MERCHCSORT_HW",
                "target_type": "Merchantable",
                "target": 31,
                "disturbance_type_id": 73,
                "efficiency": 0.99},
            mock_state_variables="mock_state_vars",
            mock_pools="pools",
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=31,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=expected_sort_value,
            efficiency=0.99,
            eligible="eligible"
        )

    def test_create_sit_event_target_random_sort_merch_target(self):

        mock_pools = pd.DataFrame({"a": [12, 3, 4, 5]})

        def mock_random_gen(n_values):
            return [1] * n_values

        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3],
            DisturbanceSoftProduction=[1, 1, 1, 1],
            DisturbanceHardProduction=[1, 1, 1, 1],
            DisturbanceDOMProduction=[1, 1, 1, 1])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 43)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(list(cbm_vars.pools.a) == [12, 3, 4, 5])
            return mock_production

        get_test_function(
            mock_sit_event_row={
                "sort_type": "RANDOMSORT",
                "target_type": "Merchantable",
                "target": 31,
                "disturbance_type_id": 43,
                "efficiency": 0.99},
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=mock_random_gen,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=31,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_random_gen(mock_pools.shape[0]),
            efficiency=0.99,
            eligible="eligible"
        )

    def test_create_sit_event_target_total_stem_snag_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 15)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.SoftwoodStemSnag == [1, 2, 3])
            self.assertTrue(cbm_vars.pools.HardwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(
            SoftwoodStemSnag=[1, 2, 3],
            HardwoodStemSnag=[1, 2, 3])

        get_test_function(
            mock_sit_event_row={
                "sort_type": "TOTALSTEMSNAG",
                "target_type": "Merchantable",
                "target": 37,
                "disturbance_type_id": 15,
                "efficiency": 1.0},
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=37,
            disturbance_production=mock_production,
            inventory="inventory",
            # simply confirm the '+' operator is used on the correct pools
            sort_value=(
                mock_pools.SoftwoodStemSnag +
                mock_pools.HardwoodStemSnag),
            efficiency=1.0,
            eligible="eligible"
        )

    def test_create_sit_event_target_sw_stem_snag_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 57)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.SoftwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(
            SoftwoodStemSnag=[1, 2, 3])

        get_test_function(
            mock_sit_event_row={
                "sort_type": "SWSTEMSNAG",
                "target_type": "Merchantable",
                "target": 47,
                "disturbance_type_id": 57,
                "efficiency": 1.1},
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=47,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_pools.SoftwoodStemSnag,
            efficiency=1.1,
            eligible="eligible"
        )

    def test_create_sit_event_target_hw_stem_snag_sort_merch_target(self):
        mock_production = SimpleNamespace(
            Total=[3, 3, 3, 3])

        def mock_disturbance_production_func(cbm_vars, disturbance_type_id):
            self.assertTrue(disturbance_type_id == 9)
            self.assertTrue(cbm_vars.inventory == "inventory")
            self.assertTrue(cbm_vars.pools.HardwoodStemSnag == [1, 2, 3])
            return mock_production

        mock_pools = SimpleNamespace(
            HardwoodStemSnag=[1, 2, 3])

        get_test_function(
            mock_sit_event_row={
                "sort_type": "HWSTEMSNAG",
                "target_type": "Merchantable",
                "target": 97,
                "disturbance_type_id": 9,
                "efficiency": 2.1},
            mock_state_variables="mock_state_vars",
            mock_pools=mock_pools,
            mock_random_generator=None,
            mock_disturbance_production_func=mock_disturbance_production_func
        ).sorted_merch_target.assert_called_once_with(
            carbon_target=97,
            disturbance_production=mock_production,
            inventory="inventory",
            sort_value=mock_pools.HardwoodStemSnag,
            efficiency=2.1,
            eligible="eligible"
        )

    def test_create_sit_event_target_proportion_sort_proportion_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SORT_BY_HW_AGE",
                "target_type": "Area",
                "target": 100},
            mock_state_variables=SimpleNamespace(age=[10, 2, 30]),
            mock_pools="pools",
            mock_random_generator=None
        ).sorted_area_target.assert_called_once_with(
            area_target_value=100,
            sort_value=[10, 2, 30],
            inventory="inventory",
            eligible="eligible"
        )

    def test_create_sit_event_target_svoid_sort_proportion_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Proportion",
                "target": 100,
                "spatial_reference": 1000},
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None
        ).spatially_indexed_target.assert_called_once_with(
            identifier=1000,
            inventory="inventory"
        )

    def test_create_sit_event_target_svoid_sort_merch_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Merchantable",
                "target": 10,
                "spatial_reference": 4000},
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None
        ).spatially_indexed_target.assert_called_once_with(
            identifier=4000,
            inventory="inventory"
        )

    def test_create_sit_event_target_svoid_sort_area_target(self):
        get_test_function(
            mock_sit_event_row={
                "sort_type": "SVOID",
                "target_type": "Area",
                "target": 130,
                "spatial_reference": 1050},
            mock_state_variables="inventory",
            mock_pools="pools",
            mock_random_generator=None
        ).spatially_indexed_target.assert_called_once_with(
            identifier=1050,
            inventory="inventory"
        )
