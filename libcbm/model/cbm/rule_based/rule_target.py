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
        flux["DisturbanceDOMProduction"]).multiply(
            inventory.area)

    if production_c < carbon_target:
        # unrealized carbon target

    disturbed_inventory = inventory.copy()
    disturbed_inventory["production_c"] = production_c
