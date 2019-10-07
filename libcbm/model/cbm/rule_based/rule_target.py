import pandas as pd
import numpy as np
from libcbm.model.cbm import cbm_variables


def disturbance_flux_target(cbm, carbon_target, pools, inventory,
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
        "disturbance_type": np.ones(n_stands)*disturbance_type})
    cbm.model_functions.GetDisturbanceOps(
        disturbance_op, inventory, disturbance_type)
    flux = cbm_variables.initialize_flux(n_stands, flux_indicator_codes)
    cbm.compute_functions.ComputeFlux(
        [disturbance_op], [disturbance_op_process_id],
        pools.copy(), flux, enabled=None)

    # computes C harvested by applying the disturbance matrix to the specified
    # carbon pools
    production_c = (
        flux["DisturbanceSoftProduction"] +
        flux["DisturbanceHardProduction"] +
        flux["DisturbanceDOMProduction"]
    ).multiply(inventory.area, axis=0)

    target_var = production_c
    target = carbon_target
    if target_var == 0 and target > 0:
        # unrealized target
        return

    disturbed = inventory.copy()
    disturbed["target_var"] = target_var
    disturbed = disturbed.sort_values(by="target_var", ascending=False)
    #filter out records that produced nothing towards the target
    disturbed = disturbed.loc[disturbed.target_var > 0]
    if disturbed.shape[0] == 0:
        # error, there are no records contributing to the target
        return
    # compute the cumulative sums of the target var to compare versus the
    # target value
    disturbed["target_var_sums"] = disturbed["target_var"].cumsum()
    disturbed = disturbed.reset_index()

    fully_disturbed_records = disturbed[
        disturbed.target_var_sums <= target]
    remaining_target = target
    if fully_disturbed_records.shape[0] > 0:
        remaining_target = target - fully_disturbed_records["target_var_sums"].max()

    partial_disturb = disturbed[disturbed.target_var_sums > target]
    if partial_disturb.shape[0] == 0 and remaining_target > 0:
        # unrealized target
        return

    split_record = partial_disturb.iloc[[0]]

    result = pd.DataFrame({
        "disturbed_indices": pd.concat(
            (fully_disturbed_records["index"], split_record["index"])
        ),
        "area_proportions": pd.concat(
            (pd.Series(np.ones(len(fully_disturbed_records["index"]))),
             split_record.area * remaining_target / split_record.production_c )
        )})