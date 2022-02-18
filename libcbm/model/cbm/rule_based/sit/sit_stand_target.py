# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Callable
from typing import Union
from libcbm.input.sit import sit_disturbance_event_parser
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm.rule_based.rule_target import RuleTargetResult
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.cbm_variables import Series
from libcbm.model.cbm.cbm_variables import DataFrame


def _is_production_sort(sit_event_row: dict) -> bool:
    production_sorts = ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]
    if sit_event_row["sort_type"] in production_sorts:
        # sorted by the production
        return True
    return False


def _is_production_based(sit_event_row: dict) -> bool:
    """Returns true if the specified disturbance event requires computation
    of production via disturbance matrix

    Args:
        sit_event_row (dict): a row dict from parsed SIT disturbance events
    """
    if sit_event_row["sort_type"] == "SVOID":
        # spatially explicit cannot be a merch production target
        return False
    if _is_production_sort(sit_event_row):
        return True
    if (
        sit_event_row["target_type"]
        == sit_disturbance_event_parser.get_target_types()["M"]
    ):
        # the production is the target variable
        return True
    return False


def _get_production_sort_value(
    sort_type: str, production: DataFrame, cbm_vars: CBMVariables
) -> Series:
    if sum(production.Total) == 0:
        return cbm_vars.pools.SoftwoodMerch + cbm_vars.pools.HardwoodMerch
    if sort_type == "MERCHCSORT_TOTAL":
        return production.Total
    elif sort_type == "MERCHCSORT_SW":
        return (
            production.DisturbanceSoftProduction
            + production.DisturbanceDOMProduction
        )
    elif sort_type == "MERCHCSORT_HW":
        return (
            production.DisturbanceHardProduction
            + production.DisturbanceDOMProduction
        )
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not a production sort"
        )


def _get_sort_value(
    sort_type: str,
    cbm_vars: CBMVariables,
    random_generator: Callable[[int], Series],
) -> Series:
    if sort_type == "SORT_BY_SW_AGE" or sort_type == "SORT_BY_HW_AGE":
        return cbm_vars.state.age
    elif sort_type == "TOTALSTEMSNAG":
        return (
            cbm_vars.pools.SoftwoodStemSnag + cbm_vars.pools.HardwoodStemSnag
        )
    elif sort_type == "SWSTEMSNAG":
        return cbm_vars.pools.SoftwoodStemSnag
    elif sort_type == "HWSTEMSNAG":
        return cbm_vars.pools.HardwoodStemSnag
    elif sort_type == "RANDOMSORT":
        return random_generator(cbm_vars.pools.shape[0])
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not sort value"
        )


def create_sit_event_target_factory(
    sit_event_row: dict,
    disturbance_production_func: Callable[
        [CBMVariables, Union[int, Series]], Series
    ],
    random_generator: Callable[[int], Series],
):
    def factory(cbm_vars: CBMVariables, eligible: Series):
        return create_sit_event_target(
            sit_event_row,
            cbm_vars,
            disturbance_production_func,
            eligible,
            random_generator,
        )

    return factory


def create_sit_event_target(
    sit_event_row: dict,
    cbm_vars: CBMVariables,
    disturbance_production_func: Callable[
        [CBMVariables, Union[int, Series]], Series
    ],
    eligible: Series,
    random_generator: Callable[[int], Series],
) -> RuleTargetResult:

    sort = sit_event_row["sort_type"]
    target_type = sit_event_row["target_type"]
    target = sit_event_row["target"]
    target_types = sit_disturbance_event_parser.get_target_types()
    area_target_type = target_types["A"]
    merchantable_target_type = target_types["M"]
    proportional_target_type = target_types["P"]
    non_sorted = ["SVOID", "PROPORTION_OF_EVERY_RECORD"]
    if _is_production_based(sit_event_row):
        production = disturbance_production_func(
            cbm_vars, sit_event_row["disturbance_type_id"]
        )
    rule_target_result = None
    if target_type == area_target_type and sort not in non_sorted:
        if _is_production_sort(sit_event_row):
            rule_target_result = rule_target.sorted_area_target(
                area_target_value=target,
                sort_value=_get_production_sort_value(
                    sort, production, cbm_vars.pools
                ),
                inventory=cbm_vars.inventory,
                eligible=eligible,
            )
        else:
            rule_target_result = rule_target.sorted_area_target(
                area_target_value=target,
                sort_value=_get_sort_value(sort, cbm_vars, random_generator),
                inventory=cbm_vars.inventory,
                eligible=eligible,
            )
    elif target_type == merchantable_target_type and sort not in non_sorted:
        if _is_production_sort(sit_event_row):
            rule_target_result = rule_target.sorted_merch_target(
                carbon_target=target,
                disturbance_production=production,
                inventory=cbm_vars.inventory,
                sort_value=_get_production_sort_value(
                    sort, production, cbm_vars.pools
                ),
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
            )
        else:
            rule_target_result = rule_target.sorted_merch_target(
                carbon_target=target,
                disturbance_production=production,
                inventory=cbm_vars.inventory,
                sort_value=_get_sort_value(
                    sort, cbm_vars, random_generator
                ),
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
            )
    elif sort == "SVOID":
        rule_target_result = rule_target.spatially_indexed_target(
            identifier=sit_event_row["spatial_reference"],
            inventory=cbm_vars.inventory,
        )
    elif target_type == proportional_target_type:
        if sort == "PROPORTION_OF_EVERY_RECORD":
            rule_target_result = rule_target.proportion_sort_proportion_target(
                proportion_target=target,
                inventory=cbm_vars.inventory,
                eligible=eligible,
            )
    elif sort == "PROPORTION_OF_EVERY_RECORD":
        if target_type == area_target_type:
            rule_target_result = rule_target.proportion_area_target(
                area_target_value=target,
                inventory=cbm_vars.inventory,
                eligible=eligible,
            )
        elif target_type == merchantable_target_type:
            rule_target_result = rule_target.proportion_merch_target(
                carbon_target=target,
                disturbance_production=production.Total,
                inventory=cbm_vars.inventory,
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
            )
    if rule_target_result is None:
        raise ValueError(
            f"specified sort ({sort}), target_type ({target_type}) "
            "is not valid"
        )
    return rule_target_result
