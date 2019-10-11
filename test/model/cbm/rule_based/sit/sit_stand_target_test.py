import unittest
from types import SimpleNamespace
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based.sit import sit_stand_target
from libcbm.model.cbm.rule_based import rule_target


class SITStandTargetTest(unittest.TestCase):

    def test_create_sit_event_target(self):

        mock_rule_target = Mock(spec=rule_target)

        mock_disturbance_production_func = Mock()
        mock_unrealized = "on_unrealized"
        create_target = sit_stand_target.create_sit_event_target_factory(
            rule_target=mock_rule_target,
            sit_event_row={
                "sort_type": "SORT_BY_SW_AGE",
                "target_type": "Area",
                "target": 100,
                "disturbance_type": "fire"
            },
            disturbance_production_func=mock_disturbance_production_func,
            on_unrealized=mock_unrealized)

        mock_state_variables = SimpleNamespace(age=[10, 2, 30])
        create_target(
            pools=pd.DataFrame({"a": [1, 2, 3]}),
            inventory="inventory",
            state_variables=mock_state_variables)

        mock_rule_target.sorted_area_target.assert_called_once_with(
            area_target_value=100,
            sort_value=mock_state_variables.age,
            inventory="inventory",
            on_unrealized=mock_unrealized
        )


