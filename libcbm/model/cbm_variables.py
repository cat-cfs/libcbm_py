import pandas as pd
import numpy as np


def initialize(inventory, n_stands, n_pools, n_flux_indicators, n_classifiers):
    """Format and allocate variable compatible with simulation
    functions in libcbm.model.cbm

    Arguments:
        inventory {pandas.DataFrame} -- A dataframe with n rows equal to the
            value of the n_stands parameter, contains constant values which
            define the initial state of CBM simulation
        n_stands {int} -- the number of stands in the resulting dataset
        n_pools {int} -- the number of pools
        n_flux_indicators {int} -- the number of flux indicators
        n_classifiers {int} -- the number of inventory classifiers

    Returns:
        dict -- dictionary containing numpy and DataFrame variables
    """
    if not inventory:
        # if not user specified, allocate inventory with default values
        inventory = pd.DataFrame({
            # simulation constant variables
            "area": np.ones(n_stands, dtype=np.int32),
            "inventory_age": np.zeros(n_stands, dtype=np.int32),
            "spatial_units": np.zeros(n_stands, dtype=np.int32),
            "afforestation_pre_type_id": np.zeros(n_stands, dtype=np.int32),
            "land_class": np.ones(n_stands, dtype=np.int32),
            "historic_disturbance_type": np.zeros(n_stands, dtype=np.int32),
            "last_pass_disturbance_type": np.zeros(n_stands, dtype=np.int32),
            "delay": np.zeros(n_stands, dtype=np.int32),
        })
    working_variables = {
        "age": np.zeros(n_stands, dtype=np.int32),
        "last_disturbance_type": np.zeros(n_stands, dtype=np.int32),
        "time_since_last_disturbance": np.zeros(n_stands, dtype=np.int32),
        "time_since_land_class_change": np.zeros(n_stands, dtype=np.int32),
        "growth_multipliers": np.ones(n_stands, dtype=np.float),
        "regeneration_delay": np.zeros(n_stands, dtype=np.int32),
        "disturbance_types": np.zeros(n_stands, dtype=np.int32),
        "transition_rules": np.zeros(n_stands, dtype=np.int32),
        "growth_enabled": np.zeros(n_stands, dtype=np.int32),
        "enabled": np.ones(n_stands, dtype=np.int32)
    }
    classifiers = np.zeros((n_stands, n_classifiers))
    pools = np.zeros((n_stands, n_pools))
    flux = np.zeros((n_stands, n_flux_indicators))
    return {
        "inventory": inventory,
        "working_variables": working_variables,
        "classifiers": classifiers,
        "pools": pools,
        "flux": flux
    }
