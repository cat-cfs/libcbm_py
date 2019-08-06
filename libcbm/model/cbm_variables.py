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
    """Create a dataframe for storing CBM pools

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Arguments:
        n_stands {int} -- The number of stands, and therefore rows in the
            resulting dataframe.
        pool_codes {list} -- a list of pool names, which are used as column
            labels in the resulting dataframe

    Returns:
        pandas.DataFrame -- A dataframe for storing CBM pools
    """
    return pd.DataFrame(
        data=np.zeros(n_stands, len(pool_codes)),
        columns=pool_codes)


def initialize_flux(n_stands, flux_indicator_codes):
    """Create a dataframe for storing CBM flux indicator values

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Arguments:
        n_stands {int} -- The number of stands, and therefore rows in the
            resulting dataframe.
        flux_indicator_codes {list} -- a list of flux indicator names, which
            are used as column labels in the resulting dataframe

    Returns:
        pandas.DataFrame -- A dataframe for storing CBM flux indicators
    """
    return pd.DataFrame(
        data=np.zeros(n_stands, len(flux_indicator_codes)),
        columns=flux_indicator_codes)


def initialize_spinup_parameters(n_stands, return_interval=None,
                                 min_rotations=None, max_rotations=None,
                                 mean_annual_temp=None):
    """Create spinup parameters as a collection of variable vectors

    The variables here are all of length N stands and are row-aligned with
    all other vectors and dataframes using this convention.

    Each keyword argument is optional, and if unspecified, libcbm will use a
    default for the corresponding parameter drawn from cbm_defaults.  These
    parameters are available here to override those default values on a
    per-stand basis.

    If a scalar value is provided to any of the parameters, that value will
    be filled in the resulting vector.

    Arguments:
        n_stands {int} -- The length of each of the resulting variables
            vectors returned by this function.

    Keyword Arguments:
        return_interval {numpy.ndarray} -- The number of years between
            historical disturbances in the spinup function. (default: {None})
        min_rotations {numpy.ndarray} -- The minimum number of historical
            rotations to perform. (default: {None})
        max_rotations {numpy.ndarray} -- The maximum number
            of historical rotations to perform. (default: {None})
        mean_annual_temp {numpy.ndarray} -- The mean annual temperature used
            in the spinup procedure (default: {None})

    Returns:
        object -- Returns an object with properties to access each of the
            spinup parameters
    """
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
    """Creates a collection of vectors used as working/state variables for
    the spinup routine.

    The variables here are all of length N stands and are row-aligned with
    all other vectors and dataframes using this convention.

    Arguments:
        n_stands {int} -- The number of stands
        pools {pandas.DataFrame} -- a dataframe containing pools of dimension
            n_stands by n_pools.

    Returns:
        object -- an object with properties to access working variables
        needed by the spinup routine.
    """

    if len(pools.index) != n_stands:
        raise ValueError(
            "Number of pools does not match number of rows "
            "in provided pools dataframe.")
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
                              transition_id=0, mean_annual_temp=None):
    """Create CBM parameters as a collection of variable vectors

    The variables here are all of length N stands and are row-aligned with
    all other vectors and dataframes using this convention.

    The mean_annual temperature keyword argument is optional, and if
    unspecified, libcbm will use a default for the corresponding parameter
    drawn from cbm_defaults.

    If a scalar value is provided to any of the parameters, that value will
    be filled in the resulting vector.

    Arguments:
        n_stands {int} -- The number of stands

    Keyword Arguments:
        disturbance_type {numpy.ndarray} -- The disturbance type id which
            references the disturbance types defined in the libCBM
            configuration.  By convention, a negative or 0 value indicates
            no disturbance (default: {0})
        transition_id {numpy.ndarray} -- The transition id which references
            the transition rules defined in the libCBM configuration.  By
            convention, a negative or 0 value indicates no transition
            (default: {0})
        mean_annual_temp {numpy.ndarray} -- A value, in degrees Celsius,
            that defines this timestep's mean annual temperature for each
            stand. (default: {None})

    Returns:
        object -- an object with properties for each cbm parameter used by
            the cbm step function.
    """
    parameters = SimpleNamespace()
    parameters.disturbance_type = promote_scalar(
        disturbance_type, n_stands, dtype=np.int32)
    parameters.transition_rule_id = promote_scalar(
        disturbance_type, n_stands, dtype=np.int32)
    parameters.mean_annual_temp = promote_scalar(
        disturbance_type, n_stands, dtype=np.float)
    return parameters


def initialize_cbm_state_variables(n_stands):
    """Creates a pandas dataframe containing state variables used by CBM
    functions at simulation runtime, with default initial values.

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Arguments:
        n_stands {int} -- the number of rows in the resulting dataframe.

    Returns:
        pandas.DataFrame -- a dataframe containing the CBM state variables.
    """
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


def initialize_cbm_variables(pools, flux, state):
    """Gathers CBM variables into a common object, and performs some validation.

    All dataframes here have 1 row for each stand and are row-aligned with
    all other vectors and dataframes using this convention.

    Arguments:
        pools {pandas.DataFrame} -- contains pool values for CBM simulation
        flux {pandas.DataFrame} -- contains pool values for CBM simulation
        state {pandas.DataFrame} -- contains state values for CBM simulation

    Returns:
        object -- an object with properties to access the pool, flux and
        state simulation variables used by CBM.
    """
    variables = SimpleNamespace()

    variables.pools = pools
    variables.flux = flux
    variables.state = state
    return variables


def initialize_inventory(classifiers, inventory):
    """creates input for libcbm.model.CBM functions


    Arguments:
        classifiers {pandas.DataFrame} -- dataframe of inventory classifier
            sets. Column names are the name of the classifiers, and values
            are the ids for each classifier value associated with the
            inventory at each row.
        inventory {pandas.DataFrame} -- data defining the inventory

    Raises:
        ValueError: Raised if the number of rows for classifiers and
        inventory are not the same.

    Returns:
        object -- an object containing the inventory and classifier
        dataframes.
    """
    n_stands = len(inventory.index)
    if not len(classifiers.index) == n_stands:
        raise ValueError(
            ("number of inventory records: {inv} does not match number of "
             "classifier sets: {c_sets}").format(
                inv=n_stands, c_sets=len(classifiers.index)))
    i = SimpleNamespace()
    i.classifiers = classifiers
    i.inventory = inventory
    return i
