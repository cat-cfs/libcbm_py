import pandas as pd
import numpy as np
from libcbm.model.cbm import cbm_variables


def sorted_disturbance_target(target_var, sort_var, target):

    if target_var.sum() <= 0 and target > 0:
        # unrealized target
        raise ValueError("unrealized target")

    disturbed = pd.DataFrame({
        "target_var": target_var,
        "sort_var": sort_var})
    disturbed = disturbed.sort_values(by="sort_var", ascending=False)
    # filter out records that produced nothing towards the target
    disturbed = disturbed.loc[disturbed.target_var > 0]
    if disturbed.shape[0] == 0:
        # error, there are no records contributing to the target
        raise ValueError("no target values greater that zero")
    # compute the cumulative sums of the target var to compare versus the
    # target value
    disturbed["target_var_sums"] = disturbed["target_var"].cumsum()
    disturbed = disturbed.reset_index()

    fully_disturbed_records = disturbed[
        disturbed.target_var_sums <= target]
    remaining_target = target
    if fully_disturbed_records.shape[0] > 0:
        remaining_target = \
            target - fully_disturbed_records["target_var_sums"].max()

    partial_disturb = disturbed[disturbed.target_var_sums > target]
    if partial_disturb.shape[0] == 0 and remaining_target > 0:
        # unrealized target
        raise ValueError("unrealized target")

    result = pd.DataFrame({
        "disturbed_indices": fully_disturbed_records["index"],
        "area_proportions":  np.ones(len(fully_disturbed_records["index"]))})

    if partial_disturb.shape[0] > 0:
        # for merch C and area targets a final record is split to meet target
        # exactly
        split_record = partial_disturb.iloc[[0]]
        result = result.append(
            pd.DataFrame({
                "disturbed_indices": split_record["index"],
                "area_proportions": remaining_target / split_record.target_var
        }))

    return result


def area_target(area_target_value, sort_value, inventory):
    return sorted_disturbance_target(
        target_var=inventory.area,
        sort_var=sort_value,
        target=area_target_value)


def merch_target(carbon_target, disturbance_production, inventory, sort_value,
                 efficiency):

    production_c = disturbance_production.Total * inventory.area * efficiency
    result = sorted_disturbance_target(
        target_var=production_c,
        sort_var=sort_value,
        target=carbon_target)
    result.area_proportions = result.area_proportions * efficiency
    return result


def compute_disturbance_production(cbm, pools, inventory,
                                   disturbance_type, flux_indicator_codes):

    # compute the flux based on the specified disturbance type

    # this is by convention in the cbm_defaults database
    disturbance_op_process_id = 3

    # The number of stands is the number of rows in the inventory table.
    # The set of inventory here is assumed to be the eligible for disturbance
    # filtered subset of records
    n_stands = inventory.shape[0]

    # allocate space for computing the Carbon flows
    disturbance_op = cbm.compute_functions.AllocateOp(n_stands)

    # set the disturbance type for all records
    disturbance_type = pd.DataFrame({
        "disturbance_type": np.ones(n_stands) * disturbance_type})
    cbm.model_functions.GetDisturbanceOps(
        disturbance_op, inventory, disturbance_type)
    flux = cbm_variables.initialize_flux(n_stands, flux_indicator_codes)
    cbm.compute_functions.ComputeFlux(
        [disturbance_op], [disturbance_op_process_id],
        pools.copy(), flux, enabled=None)

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
