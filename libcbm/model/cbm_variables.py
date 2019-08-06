import pandas as pd
import numpy as np
from types import SimpleNamespace


def promote_scalar(value, size, dtype):
    """If the specified value is scalar promote it to a numpy array filled
    with the scalar value, and otherwise return the value.  This is purely
    a helper function to allow scalar parameters for certain vector
    functions

    Arguments:
        value {ndarray} or {number} or {None} -- value to promote
        size {int} -- the length of the resulting vector if promotion
        occurs
        dtype {object} -- object used to define the type of the resulting
        vector if promotion occurs

    Returns:
        ndarray or None -- returns either the original value, a promoted
        scalar or None depending on the specified values
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    else:
        return np.ones(size, dtype=dtype) * value


def initialize_pools(n_stands, pool_codes):
    return pd.DataFrame(
        data=np.zeros(n_stands, len(pool_codes)),
        columns=pool_codes)


def initialize_flux_indicators(n_stands, flux_indicator_codes):
    return pd.DataFrame(
        data=np.zeros(n_stands, len(flux_indicator_codes)),
        columns=flux_indicator_codes)


def initialze_spinup_parameters(n_stands, return_interval=None,
                                min_rotations=None, max_rotations=None,
                                mean_annual_temp=None):
    parameters = SimpleNamespace()
    parameters.return_interval = promote_scalar(
        return_interval, n_stands, dtype=np.int32)
    parameters.min_rotations = promote_scalar(
        min_rotations, n_stands, dtype=np.int32)
    parameters.max_rotations = promote_scalar(
        max_rotations, n_stands, dtype=np.int32)
    parameters.mean_annual_temp = promote_scalar(
        mean_annual_temp, n_stands, dtype=np.float)
    return parameters


def initialize_spinup_variables(n_stands, pools):
    variables = SimpleNamespace()
    variables.spinup_state = np.zeros(n_stands, dtype=np.uint32)
    variables.slowPools = np.zeros(n_stands, dtype=np.float)
    variables.disturbance_types = np.zeros(n_stands, dtype=np.int32)
    variables.rotation = np.zeros(n_stands, dtype=np.int32)
    variables.step = np.zeros(n_stands, dtype=np.int32)
    variables.lastRotationSlowC = np.zeros(n_stands, dtype=np.float)
    variables.enabled = np.ones(n_stands, dtype=np.int32)
    variables.age = np.zeros(n_stands, dtype=np.int32)
    variables.growth_enabled = np.ones(n_stands, dtype=np.int32)
    variables.pools = pools
    return variables


def initialize_cbm_variables(n_stands, pools, flux):
    variables = SimpleNamespace()

    return variables


def initialize_classifiers(n_stands, classifier_names):
    pd.DataFrame(
        data=np.zeros(n_stands, len(classifier_names)),
        columns=classifier_names)


def initialize_inventory(n_stands, classifiers, inventory):
    i = SimpleNamespace()
    i.classifiers = classifiers

    required_cols = [
        "spatial_unit", "age", "spatial_units",
        "afforestation_pre_type_id", "land_class",
        "historic_disturbance_type", "last_pass_disturbance_type",
        "delay"]
    # validate the inventory columns, for other functions to operate, at least
    # the above columns are required
    cols = list(inventory.columns.values)
    missing_cols = []
    for c in required_cols:
        if c not in cols:
            missing_cols.append(c)
    if len(missing_cols) > 0:
        raise ValueError(
            "columns missing from inventory: {}".format(
                ", ".join(missing_cols)
            ))
    i.inventory = inventory


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
            "area": np.ones(n_stands, dtype=np.float),
            "age": np.zeros(n_stands, dtype=np.int32),
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
