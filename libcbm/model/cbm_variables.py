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


def append_simulation_result(simulation_result, timestep_data, timestep):
    """Append the specified timestep data to a simulation result spanning
        multiple time steps

    Arguments:
        simulation_result {pandas.DataFrame} -- a dataframe storing the
            simulation results.  If this parameter is None a new dataframe
            will be created with the single timestep as the contents.
        timestep_data {pandas.DataFrame} -- a dataframe storing a single
            timestep result
        timestep {int} -- an integer which will be added to the data appended
            to the larger simulation result in the "timestep" column

    Returns:
        pandas.DataFrame -- The simulation result with the specified timestep
            data appended
    """
    ts = timestep_data.copy()
    ts.reset_index()  # adds a column "index"
    ts.insert(loc=0, column="timestep", value=timestep)
    if simulation_result is None:
        simulation_result = ts
    else:
        simulation_result.append(ts)
    return simulation_result


def initialize_pools(n_stands, pool_codes):
    return pd.DataFrame(
        data=np.zeros(n_stands, len(pool_codes)),
        columns=pool_codes)


def initialize_flux(n_stands, flux_indicator_codes):
    return pd.DataFrame(
        data=np.zeros(n_stands, len(flux_indicator_codes)),
        columns=flux_indicator_codes)


def initialize_spinup_parameters(n_stands, return_interval=None,
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


def initialize_cbm_parameters(n_stands, disturbance_type=0,
                              transition_rule_id=0, mean_annual_temp=None):
    parameters = SimpleNamespace()
    parameters.disturbance_type = promote_scalar(
        disturbance_type, n_stands, dtype=np.int32)
    parameters.transition_rule_id = promote_scalar(
        disturbance_type, n_stands, dtype=np.int32)
    parameters.mean_annual_temp = promote_scalar(
        disturbance_type, n_stands, dtype=np.float)
    return parameters


def initialize_cbm_state_variables(n_stands):
    return pd.DataFrame({
        "last_disturbance_type": np.zeros(n_stands, dtype=np.int32),
        "time_since_last_disturbance": np.zeros(n_stands, dtype=np.int32),
        "time_since_land_class_change": np.zeros(n_stands, dtype=np.int32),
        "growth_enabled": np.zeros(n_stands, dtype=np.int32),
        "enabled": np.ones(n_stands, dtype=np.int32),
        "land_class": np.ones(n_stands, dtype=np.int32),
        "age": np.zeros(n_stands, dtype=np.int32),
        "growth_multiplier": np.ones(n_stands, dtype=np.float),
        "regeneration_delay": np.zeros(n_stands, dtype=np.int32)
    })


def initialize_cbm_variables(n_stands, pools, flux, state):
    variables = SimpleNamespace()
    variables.pools = pools
    variables.flux = flux
    variables.state = state
    return variables


def initialize_classifiers(n_stands, classifier_names):
    pd.DataFrame(
        data=np.zeros(n_stands, len(classifier_names)),
        columns=classifier_names)


def initialize_inventory(classifiers, inventory):
    """creates input for libcbm.model.CBM functions

    Arguments:
        classifiers {pandas.DataFrame} -- [description]
        inventory {pandas.DataFrame} -- [description]

    Raises:
        ValueError: [description]

    Returns:
        [type] -- [description]
    """
    n_stands = len(inventory.index)
    if not len(classifiers.index) == n_stands:
        raise ValueError(
            ("number of inventory records: {inv} does not match number of "
             "classifier sets: {c_sets}").format(
                inv=n_stands, c_sets=len(classifiers.index)))
    i = SimpleNamespace()
    i.classifiers = classifiers

    required_cols = [
        "spatial_unit", "age", "afforestation_pre_type_id", "land_class",
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
    return i
