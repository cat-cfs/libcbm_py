from typing import Callable
from libcbm.input.sit import sit_disturbance_event_parser
from libcbm.storage.series import Series
from libcbm.storage.dataframe import DataFrame
from libcbm.model.cbm.cbm_variables import CBMVariables


def get_sort_value(
    sort_type: str,
    cbm_vars: CBMVariables,
    random_generator: Callable[[int], Series],
) -> Series:
    if sort_type == "SORT_BY_SW_AGE" or sort_type == "SORT_BY_HW_AGE":
        return cbm_vars.state["age"]
    elif sort_type == "TOTALSTEMSNAG":
        return (
            cbm_vars.pools["SoftwoodStemSnag"]
            + cbm_vars.pools["HardwoodStemSnag"]
        )
    elif sort_type == "SWSTEMSNAG":
        return cbm_vars.pools["SoftwoodStemSnag"]
    elif sort_type == "HWSTEMSNAG":
        return cbm_vars.pools["HardwoodStemSnag"]
    elif sort_type == "RANDOMSORT":
        return random_generator(cbm_vars.pools.n_rows)
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not sort value"
        )


def is_production_sort(sit_event_row: dict) -> bool:
    production_sorts = ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]
    if sit_event_row["sort_type"] in production_sorts:
        # sorted by the production
        return True
    return False


def is_production_based(sit_event_row: dict) -> bool:
    """Returns true if the specified disturbance event requires computation
    of production via disturbance matrix

    Args:
        sit_event_row (dict): a row dict from parsed SIT disturbance events
    """
    if sit_event_row["sort_type"] == "SVOID":
        # spatially explicit cannot be a merch production target
        return False
    if is_production_sort(sit_event_row):
        return True
    if (
        sit_event_row["target_type"]
        == sit_disturbance_event_parser.get_target_types()["M"]
    ):
        # the production is the target variable
        return True
    return False


def get_production_sort_value(
    sort_type: str, production: DataFrame, pools: DataFrame
) -> Series:
    if sort_type not in ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not a production sort"
        )
    if production["Total"].sum() == 0:
        return pools["SoftwoodMerch"] + pools["HardwoodMerch"]
    if sort_type == "MERCHCSORT_TOTAL":
        return production["Total"]
    elif sort_type == "MERCHCSORT_SW":
        return (
            production["DisturbanceSoftProduction"]
            + production["DisturbanceDOMProduction"]
        )
    elif sort_type == "MERCHCSORT_HW":
        return (
            production["DisturbanceHardProduction"]
            + production["DisturbanceDOMProduction"]
        )
    else:
        raise
