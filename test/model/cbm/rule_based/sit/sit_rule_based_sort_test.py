import pytest
import pandas as pd
from unittest.mock import MagicMock
from types import SimpleNamespace
from libcbm.model.cbm.rule_based.sit import sit_rule_based_sort
from libcbm.storage import dataframe
from libcbm.storage import series


def test_get_sort_value():
    pools_mock = MagicMock()
    pools_mock.__getitem__.side_effect = lambda x: dict(
        SoftwoodStemSnag=series.from_list("", [1.0]),
        HardwoodStemSnag=series.from_list("", [10.0]),
    )[x]
    pools_mock.n_rows = 999
    mock_cbm_vars = SimpleNamespace(
        state=dict(age=series.from_list("", [100])), pools=pools_mock
    )
    random_generator = MagicMock()
    random_generator.side_effect = lambda x: series.from_list("", [1, 2, 3])
    cases = {
        "SORT_BY_SW_AGE": [100.0],
        "SORT_BY_HW_AGE": [100.0],
        "TOTALSTEMSNAG": [11.0],
        "SWSTEMSNAG": [1.0],
        "HWSTEMSNAG": [10.0],
        "RANDOMSORT": [1.0, 2.0, 3.0],
    }
    for sort_type, expected_result in cases.items():
        returned_value = sit_rule_based_sort.get_sort_value(
            sort_type, mock_cbm_vars, random_generator
        )
        assert returned_value.to_numpy().dtype == "float"
        assert expected_result == returned_value.to_list()
    random_generator.assert_called_with(999)
    with pytest.raises(ValueError):
        sit_rule_based_sort.get_sort_value(
            "unsupported", mock_cbm_vars, random_generator
        )


def test_is_production_sort():
    for s in ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]:
        assert sit_rule_based_sort.is_production_sort(dict(sort_type=s))
    assert not sit_rule_based_sort.is_production_sort(
        dict(sort_type="anything else")
    )


def test_is_production_based():
    assert not sit_rule_based_sort.is_production_based(
        dict(sort_type="SVOID", target_type="anything")
    )
    for s in ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]:
        assert sit_rule_based_sort.is_production_based(
            dict(sort_type=s, target_type="anything")
        )
    assert sit_rule_based_sort.is_production_based(
        dict(sort_type="anything", target_type="Merchantable")
    )
    assert not sit_rule_based_sort.is_production_based(
        dict(sort_type="anything else", target_type="anything else")
    )


def test_get_production_sort_value_no_production():
    SoftwoodMerch = 10
    HardwoodMerch = 20

    mock_pools = pd.DataFrame(
        {"SoftwoodMerch": [SoftwoodMerch], "HardwoodMerch": [HardwoodMerch]}
    )
    mock_production = pd.DataFrame(
        {
            "Total": [0],
            "DisturbanceSoftProduction": [0],
            "DisturbanceDOMProduction": [0],
        }
    )
    result = sit_rule_based_sort.get_production_sort_value(
        sort_type="MERCHCSORT_SW",
        production=dataframe.from_pandas(mock_production),
        pools=dataframe.from_pandas(mock_pools),
    )
    assert result.sum() == SoftwoodMerch + HardwoodMerch


def test_get_production_sort_values():
    SoftwoodMerch = None
    HardwoodMerch = None
    Total = 50
    DisturbanceSoftProduction = 15
    DisturbanceHardProduction = 18
    DisturbanceDOMProduction = 17
    mock_pools = pd.DataFrame(
        {"SoftwoodMerch": [SoftwoodMerch], "HardwoodMerch": [HardwoodMerch]}
    )
    mock_production = pd.DataFrame(
        {
            "Total": [Total],
            "DisturbanceSoftProduction": [DisturbanceSoftProduction],
            "DisturbanceHardProduction": [DisturbanceHardProduction],
            "DisturbanceDOMProduction": [DisturbanceDOMProduction],
        }
    )
    sort_types = {
        "MERCHCSORT_TOTAL": 50,
        "MERCHCSORT_SW": DisturbanceSoftProduction + DisturbanceDOMProduction,
        "MERCHCSORT_HW": DisturbanceHardProduction + DisturbanceDOMProduction,
    }

    for sort_type, expected_value in sort_types.items():
        result = sit_rule_based_sort.get_production_sort_value(
            sort_type=sort_type,
            production=dataframe.from_pandas(mock_production),
            pools=dataframe.from_pandas(mock_pools),
        )
        assert result.sum() == expected_value


def test_error_raised_on_unspported_sort():
    with pytest.raises(ValueError):
        sit_rule_based_sort.get_production_sort_value(
            sort_type="unsupported sort type",
            production=None,
            pools=None,
        )
