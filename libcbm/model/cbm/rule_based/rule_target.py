import pandas as pd
import numpy as np
from libcbm.model.cbm import cbm_model


def spatially_indexed_target(identifier, inventory):
    """return a target for a single inventory record identified by the
    specified identifier

    Args:
        identifier (int): the integer identifier matching a single row in
            inventory.spatial_reference
        inventory (pandas.DataFrame): inventory records

    Raises:
        ValueError: the specified identifier was not present in the specified
            inventory spatial_reference column
        ValueError: the specified identifier appears 2 or more times in the
            specified inventory spatial_reference column

    Returns:
        [type]: [description]
    """
    match = inventory[inventory.spatial_reference == identifier]
    match_index = match.index
    if len(match_index) < 1:
        raise ValueError(
            "no matching value in inventory spatial_reference column for "
            f"identifier {identifier}")
    if len(match_index) > 1:
        raise ValueError(
            "multiple matching values in inventory spatial_reference column "
            f"for identifier {identifier}")
    result = pd.DataFrame({
        "target_var": [match.area],
        "sort_var": [0.0],
        "disturbed_index": [match_index[0]],
        "area_proportions":  [1.0]})
    return result


def sorted_disturbance_target(target_var, sort_var, target, eligible,
                              on_unrealized):
    """Given a target variable, a sort variable, and a cumulative
    target, produce a table of index, area proportions that will
    satisfy exactly a rule based disturbance target.

    Args:
        target_var (pd.Series): a series of values fed into an
            accumulator to satisfy the cumulative target.
        sort_var (pd.Series): a variable whose descending sort order
            defines the order in which target_var values are fed into
            the accumulator.
        target (float): the cumulative target.
        eligible (pandas.Series): boolean array indicating
            whether or not each index is eligible for this disturbance target
        on_unrealized (func): a function called when the specified parameters
            will result in an unrealized disturbance. target - sum(target_var)
            is passed as the single parameter.

    Raises:
        ValueError: specified target was less than 0
        ValueError: less than zero values are detected in target_var

    Returns:
        pandas.DataFrame: a data frame with columns:

            - disturbed_index: the zero based indices of the records that
                should be disturbed
            - area_proportion: the proportion of each disturbed index to
                disturb, 1 indicates the entire record, and < 1 indicates to
                disturb a proportion.
    """
    if target < 0:
        raise ValueError("target is less than zero")
    if (target_var < 0).any():
        raise ValueError("less than zero values detected in target_var")
    remaining_target = target
    result = pd.DataFrame()

    disturbed = pd.DataFrame({
        "target_var": target_var,
        "sort_var": sort_var})
    disturbed = disturbed[eligible]
    disturbed = disturbed.sort_values(by="sort_var", ascending=False)
    # filter out records that produced nothing towards the target
    disturbed = disturbed.loc[disturbed.target_var > 0]
    if disturbed.shape[0] == 0:
        if target > 0:
            on_unrealized(remaining_target)
        return result
    # compute the cumulative sums of the target var to compare versus the
    # target value
    disturbed["target_var_sums"] = disturbed["target_var"].cumsum()
    disturbed = disturbed.reset_index()

    fully_disturbed_records = disturbed[
        disturbed.target_var_sums <= target]

    if fully_disturbed_records.shape[0] > 0:
        remaining_target = \
            target - fully_disturbed_records["target_var_sums"].max()

    result = result.append(pd.DataFrame({
        "target_var": fully_disturbed_records["target_var"],
        "sort_var": fully_disturbed_records["sort_var"],
        "disturbed_index": fully_disturbed_records["index"],
        "area_proportions":  np.ones(len(fully_disturbed_records["index"]))}))

    partial_disturb = disturbed[disturbed.target_var_sums > target]

    if partial_disturb.shape[0] > 0 and remaining_target > 0:
        # for merch C and area targets a final record is split to meet target
        # exactly
        split_record = partial_disturb.iloc[0]
        proportion = remaining_target / split_record["target_var"]
        remaining_target = 0

        result = result.append(
            pd.DataFrame({
                "target_var": split_record["target_var"],
                "sort_var": split_record["sort_var"],
                "disturbed_index": split_record["index"],
                "area_proportions": [proportion]
            }))
    if remaining_target > 0:
        on_unrealized(remaining_target)
    return result


def proportion_area_target(area_target_value, inventory, eligible,
                           on_unrealized):
    """create a disturbance target which disturbs that proportion of all
    eligible records that such that the sum of all eligible record
    areas multiplied by the proportion equals the area target exactly.

    Args:
        area_target_value (float): the target area to disturb
        inventory (pd.DataFrame): the inventory being targetted for
            disturbance.
        eligible (pandas.Series): boolean array indicating
            whether or not each index is eligible for this disturbance target
        on_unrealized (func): a function called when the specified parameters
            will result in an unrealized disturbance.
            area_target_value - sum(inventory.area[eligible]) is passed as the
            single parameter.
    """
    raise NotImplementedError()


def sorted_area_target(area_target_value, sort_value, inventory, eligible,
                       on_unrealized):
    """create a sorted sequence of areas/proportions for meeting an area
    target exactly.

    Args:
        area_target_value (float): the target area to disturb
        sort_value (pd.Series): a sequence of values whose decending sort
            defines the order to accumulate areas.  Length must equal the
            number of rows in the specified inventory
        inventory (pd.DataFrame): the inventory being targetted for
            disturbance.
        eligible (pandas.Series): boolean array indicating
            whether or not each index is eligible for this disturbance target
        on_unrealized (func): a function called when the specified parameters
            will result in an unrealized disturbance.
            area_target_value - sum(inventory.area[eligible]) is passed as the
            single parameter.

    pandas.DataFrame: a data frame specifying the sorted disturbance event
        area target. Has the same format as the return value of
        :py:func:`sorted_disturbance_target`

    """
    if inventory.shape[0] != sort_value.shape[0]:
        raise ValueError(
            "sort_value dimension must equal number of rows in inventory")
    return sorted_disturbance_target(
        target_var=inventory.area,
        sort_var=sort_value,
        target=area_target_value,
        eligible=eligible,
        on_unrealized=on_unrealized)


def proportion_merch_target(carbon_target, disturbance_production, inventory,
                            efficiency, eligible, on_unrealized):
    raise NotImplementedError()


def sorted_merch_target(carbon_target, disturbance_production, inventory,
                        sort_value, efficiency, eligible, on_unrealized):
    """create a sorted sequence of areas/proportions for meeting a merch C
    target exactly.

    Args:
        carbon_target (float): a disturbance target in CBM mass units
            (tonnes C)
        disturbance_production (pandas.DataFrame): a table of Carbon
            generated by a disturbance events on the specified inventory.
            Used in accumulating value towards the carbon_target parameter.
            See :py:func:`compute_disturbance_production`
        inventory (pd.DataFrame): the inventory being targetted for
            disturbance.
        sort_value (pd.Series): a sequence of values whose decending sort
            defines the order to accumulate carbon mass.  Length must equal
            the number of rows in the specified inventory
        efficiency (float): reduce the disturbance production and split all
            records
        eligible (pandas.Series): boolean array indicating
            whether or not each index is eligible for this disturbance target
        on_unrealized (func): a function called when the specified parameters
            will result in an unrealized disturbance.
            carbon_target - sum(eligible production) is passed as the single
            parameter.

    Returns:
        pandas.DataFrame: a data frame specifying the sorted disturbance event
            merchantable C target. Has the same format as the return value of
            :py:func:`sorted_disturbance_target`
    """
    if inventory.shape[0] != sort_value.shape[0]:
        raise ValueError(
            "sort_value dimension must equal number of rows in inventory")
    if inventory.shape[0] != disturbance_production.shape[0]:
        raise ValueError(
            "number of disturbance_production rows must equal number of rows "
            "in inventory")
    production_c = disturbance_production.Total * inventory.area * efficiency
    result = sorted_disturbance_target(
        target_var=production_c,
        sort_var=sort_value,
        target=carbon_target,
        eligible=eligible,
        on_unrealized=on_unrealized)
    result.area_proportions = result.area_proportions * efficiency
    return result


def compute_disturbance_production(model_functions, compute_functions, pools,
                                   inventory, disturbance_type,
                                   flux, eligible):
    """Computes a series of disturbance production values based on

    Args:
        model_functions (object): Model specific functions. Used for computing
            disturbance flows based on the specified disturbance type.
        compute_functions (object): Functions for computing pool flows.
        pools (pandas.DataFrame): The current pool values.
        inventory (pandas.DataFrame): The inventory DataFrame.
        disturbance_type (int): The integer code specifying the disturbance
            type.
        flux (pandas.DataFrame): Storage for flux computation
        eligible (pandas.Series): Bit values where True specifies the index is
            eligible for the disturbance, and false the opposite. In the
            returned result False indices will be set with 0's.

    Returns:
        pandas.DataFrame: dataframe describing the C production associated
            with applying the specified disturbance type on the specified
            pools.  All columns are expressed as area density values
            in units tonnes C/ha.

            Fields:

                - DisturbanceSoftProduction: the softwood C production
                - DisturbanceSoftProduction: the hardwood C production
                - DisturbanceDOMProduction: the dead organic matter C
                    production
                - Total: the row sums of the above three values

    """
    # this is by convention in the cbm_defaults database
    disturbance_op_process_id = cbm_model.get_op_processes()["disturbance"]

    # The number of stands is the number of rows in the inventory table.
    # The set of inventory here is assumed to be the eligible for disturbance
    # filtered subset of records
    n_stands = inventory.shape[0]

    # allocate space for computing the Carbon flows
    disturbance_op = compute_functions.AllocateOp(n_stands)

    # set the disturbance type for all records
    disturbance_type = pd.DataFrame({
        "disturbance_type": np.ones(n_stands, dtype=np.int32) * disturbance_type})
    model_functions.GetDisturbanceOps(
        disturbance_op, inventory, disturbance_type)

    # compute the flux based on the specified disturbance type
    compute_functions.ComputeFlux(
        [disturbance_op], [disturbance_op_process_id],
        pools.copy(), flux, enabled=eligible)
    compute_functions.FreeOp(disturbance_op)
    # computes C harvested by applying the disturbance matrix to the specified
    # carbon pools
    return pd.DataFrame(data={
        "DisturbanceSoftProduction": flux["DisturbanceSoftProduction"],
        "DisturbanceHardProduction": flux["DisturbanceHardProduction"],
        "DisturbanceDOMProduction": flux["DisturbanceDOMProduction"],
        "Total":
            flux["DisturbanceSoftProduction"] +
            flux["DisturbanceHardProduction"] +
            flux["DisturbanceDOMProduction"]})
