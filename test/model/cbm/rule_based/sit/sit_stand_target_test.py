import unittest
from types import SimpleNamespace
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based.sit import sit_stand_target
from libcbm.model.cbm.rule_based import rule_target

def get_test_function(mock_sit_event_row, mock_state_variables, mock_pools,
                      mock_random_generator):
    mock_rule_target = Mock(spec=rule_target)

    mock_disturbance_production_func = Mock()
    mock_unrealized = "on_unrealized"
    mock_inventory = "inventory"
    mock_eligible = "eligible"
    create_target = sit_stand_target.create_sit_event_target_factory(
        rule_target=mock_rule_target,
        sit_event_row=mock_sit_event_row,
        disturbance_production_func=mock_disturbance_production_func,
        eligible=mock_eligible,
        on_unrealized=mock_unrealized,
        random_generator=mock_random_generator
        )

    create_target(
        pools=mock_pools,
        inventory=mock_inventory,
        state_variables=mock_state_variables)

    return mock_rule_target


class SITStandTargetTest(unittest.TestCase):
    """Tests for functions that compute CBM rule based disturbance event
    targets based on SIT disturbance event input.
    """

    def test_create_sit_event_target_proportion_sort_area_target(self):
        pass

    def test_create_sit_event_target_merch_total_sort_area_target(self):
        pass

    def test_create_sit_event_target_merch_sw_sort_area_target(self):
        pass

    def test_create_sit_event_target_merch_hw_sort_area_target(self):
        pass

    def test_create_sit_event_target_svoid_sort_area_target(self):
        pass

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
            eligible="eligible",
            on_unrealized="on_unrealized"
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
            # just check that the + operater was used.
            sort_value=[1, 2, 3, 4] + [5, 6, 7, 8],
            inventory="inventory",
            eligible="eligible",
            on_unrealized="on_unrealized"
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
            eligible="eligible",
            on_unrealized="on_unrealized"
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
            eligible="eligible",
            on_unrealized="on_unrealized"
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
            eligible="eligible",
            on_unrealized="on_unrealized"
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
            eligible="eligible",
            on_unrealized="on_unrealized"
        )

    def test_create_sit_event_target_proportion_sort_merch_target(self):
        pass

    def test_create_sit_event_target_merch_total_sort_merch_target(self):
        pass

    def test_create_sit_event_target_merch_sw_sort_merch_target(self):
        pass

    def test_create_sit_event_target_merch_hw_sort_merch_target(self):
        pass

    def test_create_sit_event_target_svoid_sort_merch_target(self):
        pass

    def test_create_sit_event_target_random_sort_merch_target(self):
        pass

    def test_create_sit_event_target_total_stem_snag_merch_area_target(self):
        pass

    def test_create_sit_event_target_sw_stem_snag_sort_merch_target(self):
        pass

    def test_create_sit_event_target_hw_stem_snag_sort_merch_target(self):
        pass

    def test_create_sit_event_target_proportion_sort_proportion_target(self):
        pass

    def test_create_sit_event_target_proportion_sort_svoid_target(self):
        pass