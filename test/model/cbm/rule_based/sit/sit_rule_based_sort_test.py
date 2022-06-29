import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
from libcbm.model.cbm.rule_based.sit import sit_rule_based_sort


def test_get_sort_value():

    pools_mock = MagicMock()
    pools_mock.__getitem__.side_effect = lambda x: dict(
        SoftwoodStemSnag=1, HardwoodStemSnag=10
    )[x]
    pools_mock.n_rows = 999
    mock_cbm_vars = SimpleNamespace(state=dict(age=100), pools=pools_mock)
    random_generator = MagicMock()
    random_generator.side_effect = lambda x: [1, 2, 3]
    cases = {
        "SORT_BY_SW_AGE": 100,
        "SORT_BY_HW_AGE": 100,
        "TOTALSTEMSNAG": 11,
        "SWSTEMSNAG": 1,
        "HWSTEMSNAG": 10,
        "RANDOMSORT": [1, 2, 3],
    }
    for sort_type, expected_result in cases.items():
        assert expected_result == sit_rule_based_sort.get_sort_value(
            sort_type, mock_cbm_vars, random_generator
        )
    random_generator.assert_called_with(999)
    with pytest.raises(ValueError):
        sit_rule_based_sort.get_sort_value(
            "unsupported", mock_cbm_vars, random_generator
        )

def test_is_production_sort():
    pass


def test_is_production_based():
    pass

def test_get_production_sort_value():
    pass