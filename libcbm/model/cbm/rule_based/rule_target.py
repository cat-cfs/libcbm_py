# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series


class RuleTargetResult:
    """
    Standard return value for the functions in this module.

    Args:
        target (DataFrame): dataframe containing:

            - target_var: the disturbed amount for each disturbed record
                (in target units)
            - sort_var: the variable used to sort values for disturbance
            - disturbed_index: the index of each disturbed record
            - area_proportions: the proportions for each record to disturb

        statistics (dict): dictionary describing the statistics involved in
            the disturbance target

    """

    def __init__(self, target: DataFrame, statistics: dict):
        self.target = target
        self.statistics = statistics


def spatially_indexed_target(
    identifier: int, inventory: DataFrame
) -> RuleTargetResult:
    """return a target for a single inventory record identified by the
    specified identifier

    Args:
        identifier (int): the integer identifier matching a single row in
            inventory.spatial_reference
        inventory (DataFrame): inventory records

    Raises:
        ValueError: the specified identifier was not present in the specified
            inventory spatial_reference column
        ValueError: the specified identifier appears 2 or more times in the
            specified inventory spatial_reference column

    Returns:
        RuleTargetResult: object with information on spatially indexed
            stand to disturb
    """
    match = inventory[inventory.spatial_reference == identifier]
    match_index = match.index
    if len(match_index) < 1:
        raise ValueError(
            "no matching value in inventory spatial_reference column for "
            f"identifier {identifier}"
        )
    if len(match_index) > 1:
        raise ValueError(
            "multiple matching values in inventory spatial_reference column "
            f"for identifier {identifier}"
        )
    result = DataFrame(
        {
            "target_var": [match.area],
            "sort_var": None,
            "disturbed_index": [match_index[0]],
            "area_proportions": [1.0],
        }
    )
    return RuleTargetResult(target=result, statistics=None)


def sorted_disturbance_target(
    target_var: Series, sort_var: Series, target: float, eligible: Series
) -> RuleTargetResult:
    """Given a target variable, a sort variable, and a cumulative
    target, produce a table of index, area proportions that will
    satisfy exactly a rule based disturbance target.

    Args:
        target_var (Series): a series of values fed into an
            accumulator to satisfy the cumulative target.
        sort_var (Series): a variable whose descending sort order
            defines the order in which target_var values are fed into
            the accumulator.
        target (float): the cumulative target.
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Raises:
        ValueError: specified target was less than 0
        ValueError: less than zero values are detected in target_var

    Returns:
        RuleTargetResult: object with information targeting a proportion of
            or the entirety of a subset of rows in the eligible subset
            of specified target_var.
    """
    if target < 0:
        raise ValueError("target is less than zero")
    if (target_var < 0).any():
        raise ValueError("less than zero values detected in target_var")
    remaining_target = target
    result = DataFrame()

    disturbed = DataFrame({"target_var": target_var, "sort_var": sort_var})
    disturbed = disturbed[eligible]
    disturbed = disturbed.sort_values(by="sort_var", ascending=False)
    # filter out records that produced nothing towards the target
    disturbed = disturbed.loc[disturbed.target_var > 0]
    if disturbed.shape[0] == 0:
        return RuleTargetResult(
            target=DataFrame(
                columns=[
                    "target_var",
                    "sort_var",
                    "disturbed_index",
                    "area_proportions",
                ]
            ),
            statistics={
                "total_eligible_value": disturbed["target_var"].sum(),
                "total_achieved": 0,
                "shortfall": target,
                "num_records_disturbed": 0,
                "num_splits": 0,
                "num_eligible": eligible.sum(),
            },
        )

    # compute the cumulative sums of the target var to compare versus the
    # target value
    disturbed["target_var_sums"] = disturbed["target_var"].cumsum()
    disturbed = disturbed.reset_index()

    fully_disturbed_records = disturbed[disturbed.target_var_sums <= target]

    if fully_disturbed_records.shape[0] > 0:
        remaining_target = (
            target - fully_disturbed_records["target_var_sums"].max()
        )

    result = pd.concat(
        [
            result,
            pd.DataFrame(
                {
                    "target_var": fully_disturbed_records["target_var"],
                    "sort_var": fully_disturbed_records["sort_var"],
                    "disturbed_index": fully_disturbed_records["index"],
                    "area_proportions": np.ones(
                        len(fully_disturbed_records["index"])
                    ),
                }
            ),
        ]
    )

    partial_disturb = disturbed[disturbed.target_var_sums > target]

    num_splits = 0
    if partial_disturb.shape[0] > 0 and remaining_target > 0:
        # for merch C and area targets a final record is split to meet target
        # exactly
        num_splits = 1
        split_record = partial_disturb.iloc[0]
        proportion = remaining_target / split_record["target_var"]
        remaining_target = 0

        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    {
                        "target_var": split_record["target_var"],
                        "sort_var": split_record["sort_var"],
                        "disturbed_index": int(split_record["index"]),
                        "area_proportions": [proportion],
                    }
                ),
            ]
        )

    result = result.reset_index(drop=True)

    stats = {
        "total_eligible_value": disturbed["target_var"].sum(),
        "total_achieved": target - remaining_target,
        "shortfall": remaining_target,
        "num_records_disturbed": result.shape[0],
        "num_splits": num_splits,
        "num_eligible": eligible.sum(),
    }
    return RuleTargetResult(target=result, statistics=stats)


def proportion_area_target(
    area_target_value: float, inventory: DataFrame, eligible: Series
) -> RuleTargetResult:
    """create a disturbance target which disturbs that proportion of all
    eligible records that such that the sum of all eligible record
    areas multiplied by the proportion equals the area target exactly.

    Args:
        area_target_value (float): the target area to disturb
        inventory (DataFrame): the inventory being targeted for
            disturbance.
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Returns:
        RuleTargetResult: object with information targeting a proportion of
            all records in the eligible subset of the specified inventory such
            that the sum of area of all disturbed records matches the specified
            area target.
    """
    eligible_inventory = inventory.filter(eligible)
    total_eligible_area = eligible_inventory["area"].sum()
    if total_eligible_area <= 0:
        return RuleTargetResult(
            target=DataFrame(
                columns=[
                    "target_var",
                    "sort_var",
                    "disturbed_index",
                    "area_proportions",
                ]
            ),
            statistics={
                "total_eligible_value": total_eligible_area,
                "total_achieved": 0.0,
                "shortfall": area_target_value,
                "num_records_disturbed": 0,
                "num_splits": 0,
                "num_eligible": len(eligible_inventory.index),
            },
        )
    area_proportion = area_target_value / total_eligible_area
    total_achieved = area_target_value
    if area_proportion >= 1:
        # shortfall
        area_proportion = 1.0
        total_achieved = total_eligible_area
    target = DataFrame(
        {
            "target_var": eligible_inventory.area * area_proportion,
            "sort_var": None,
            "disturbed_index": eligible_inventory.index,
            "area_proportions": area_proportion,
        }
    )

    num_splits = len(eligible_inventory.index) if area_proportion < 1.0 else 0
    return RuleTargetResult(
        target=target,
        statistics={
            "total_eligible_value": total_eligible_area,
            "total_achieved": total_achieved,
            "shortfall": area_target_value - total_achieved,
            "num_records_disturbed": len(eligible_inventory.index),
            "num_splits": num_splits,
            "num_eligible": len(eligible_inventory.index),
        },
    )


def sorted_area_target(
    area_target_value: float,
    sort_value: Series,
    inventory: DataFrame,
    eligible: Series,
) -> RuleTargetResult:
    """create a sorted sequence of areas/proportions for meeting an area
    target exactly.

    Args:
        area_target_value (float): the target area to disturb
        sort_value (Series): a sequence of values whose decending sort
            defines the order to accumulate areas.  Length must equal the
            number of rows in the specified inventory
        inventory (DataFrame): the inventory being targeted for
            disturbance.
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Returns:
        RuleTargetResult: object with information targeting a sorted subset
            of all records in the eligible subset of the specified inventory
            such that the total area disturbed matches the specified area
            target value.

    """
    if inventory.shape[0] != sort_value.shape[0]:
        raise ValueError(
            "sort_value dimension must equal number of rows in inventory"
        )
    return sorted_disturbance_target(
        target_var=inventory.area,
        sort_var=sort_value,
        target=area_target_value,
        eligible=eligible,
    )


def proportion_merch_target(
    carbon_target: float,
    disturbance_production: Series,
    inventory: DataFrame,
    efficiency: float,
    eligible: Series,
) -> RuleTargetResult:
    """create a sequence of areas/proportions for disturbance a propotion
    of all eligible stands such that the proportion * disturbance_production
    for each stand equals the carbon target exactly.

    Args:
        carbon_target (float): a disturbance target in CBM mass units
            (tonnes C)
        disturbance_production (DataFrame): a table of Carbon density
            (tonnes C/ha) generated by a disturbance events on the specified
            inventory. Used in accumulating value towards the carbon_target
            parameter.
        inventory (DataFrame): the inventory being targeted for
            disturbance.
        efficiency (float): A proportion value <= 1 multiplier
            disturbance production for each stand
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Returns:
        RuleTargetResult: object with information targeting a proportion of
            all records in the eligible subset of the specified inventory
            such that the total carbon produced matches the specified carbon
            target.
    """
    eligible_inventory = inventory.filter(eligible)
    eligible_production = disturbance_production.filter(eligible)
    production = eligible_production * eligible_inventory["area"] * efficiency
    total_production = production.sum()
    n_eligible = eligible_inventory.n_rows
    if total_production <= 0.0:
        return RuleTargetResult(
            target=DataFrame(
                columns=[
                    "target_var",
                    "sort_var",
                    "disturbed_index",
                    "area_proportions",
                ]
            ),
            statistics={
                "total_eligible_value": total_production,
                "total_achieved": 0.0,
                "shortfall": carbon_target,
                "num_records_disturbed": 0,
                "num_splits": 0,
                "num_eligible": n_eligible,
            },
        )
    proportion = carbon_target / total_production
    if proportion > 1:
        proportion = 1.0

    target = DataFrame(
        {
            "target_var": production * proportion,
            "sort_var": None,
            "disturbed_index": eligible_inventory.index,
            "area_proportions": proportion * efficiency,
        }
    )
    total_achieved = target.target_var.sum()
    return RuleTargetResult(
        target,
        statistics={
            "total_eligible_value": total_production,
            "total_achieved": total_achieved,
            "shortfall": carbon_target - total_achieved,
            "num_records_disturbed": n_eligible,
            "num_splits": n_eligible if proportion < 1.0 else 0,
            "num_eligible": n_eligible,
        },
    )


def proportion_sort_proportion_target(
    proportion_target: float, inventory: DataFrame, eligible: Series
) -> RuleTargetResult:
    """Create a rule target specifying to the given proportion of all of the
    eligible stands.

    Args:
        proportion_target (float): a proportion of each eligible inventory
            record's area to disturb
        inventory (DataFrame): the inventory being targeted for
            disturbance.
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Returns:
        RuleTargetResult: object with information targeting the specified
            proportion of all records in the eligible subset of the
            specified inventory.
    """
    if proportion_target < 0 or proportion_target > 1.0:
        raise ValueError(
            "proportion target may not be less than zero or greater than 1."
        )
    eligible_inventory = inventory.loc[eligible]

    n_disturbed = len(eligible_inventory.index)

    target = DataFrame(
        {
            "target_var": proportion_target,
            "sort_var": None,
            "disturbed_index": eligible_inventory.index,
            "area_proportions": proportion_target,
        }
    )
    return RuleTargetResult(
        target,
        statistics={
            "total_eligible_value": None,
            "total_achieved": None,
            "shortfall": None,
            "num_records_disturbed": n_disturbed,
            "num_splits": n_disturbed if proportion_target < 1.0 else 0,
            "num_eligible": n_disturbed,
        },
    )


def sorted_merch_target(
    carbon_target: float,
    disturbance_production: DataFrame,
    inventory: DataFrame,
    sort_value: Series,
    efficiency: float,
    eligible: Series,
) -> RuleTargetResult:
    """create a sorted sequence of areas/proportions for meeting a merch C
    target exactly.

    Args:
        carbon_target (float): a disturbance target in CBM mass units
            (tonnes C)
        disturbance_production (DataFrame): a table of Carbon density
            (tonnes C/ha) generated by a disturbance events on the specified
            inventory. Used in accumulating value towards the carbon_target
            parameter.
        inventory (DataFrame): the inventory being targeted for
            disturbance.
        sort_value (Series): a sequence of values whose decending sort
            defines the order to accumulate carbon mass.  Length must equal
            the number of rows in the specified inventory
        efficiency (float): reduce the disturbance production and split all
            records
        eligible (Series): boolean array indicating
            whether or not each index is eligible for this disturbance target

    Returns:
        RuleTargetResult: object with information targeting a subset of
            the eligible subset of the specified inventory such that the
            carbon_target is met.

    """
    if inventory.n_rows != sort_value.length:
        raise ValueError(
            "sort_value dimension must equal number of rows in inventory"
        )
    if inventory.n_rows != disturbance_production.n_rows:
        raise ValueError(
            "number of disturbance_production rows must equal number of rows "
            "in inventory"
        )
    production_c = (
        disturbance_production["Total"] * inventory["area"] * efficiency
    )
    result = sorted_disturbance_target(
        target_var=production_c,
        sort_var=sort_value,
        target=carbon_target,
        eligible=eligible,
    )
    result.target.area_proportions = (
        result.target.area_proportions * efficiency
    )
    return result
