import pandas as pd
import numpy as np
from types import SimpleNamespace


def promote_scalar(value, size, dtype):
    """If the specified value is scalar promote it to a numpy array filled
    with the scalar value, and otherwise return the value.  This is purely
    a helper function to allow scalar parameters for certain vector
    functions

    Args:
        value (numpy.ndarray, number, or None): value to promote
        size (int): the length of the resulting vector if promotion
            occurs
        dtype (object): object used to define the type of the resulting
            vector if promotion occurs


    Returns:
        ndarray or None: returns either the original value, a promoted
            scalar or None depending on the specified values
    """
    if value is None:
        return None
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.ones(size, dtype=dtype) * value


def append_simulation_result(simulation_result, timestep_data, timestep):
    """Append the specified timestep data to a simulation result spanning
    multiple time steps

    Args:
        simulation_result (pandas.DataFrame): a dataframe storing the
            simulation results.  If this parameter is None a new dataframe
            will be created with the single timestep as the contents.
        timestep_data (pandas.DataFrame): a dataframe storing a single
            timestep result
        timestep (int): an integer which will be added to the data appended
            to the larger simulation result in the "timestep" column

    Returns:
        pandas.DataFrame: The simulation result with the specified timestep
            data appended
    """
    ts = timestep_data.copy()
    ts.reset_index()  # adds a column "index"
    ts.insert(loc=0, column="timestep", value=timestep)
    if simulation_result is None:
        simulation_result = ts
    else:
        simulation_result = simulation_result.append(ts)
    return simulation_result


def initialize_pools(n_stands, pool_codes):
    """Create a dataframe for storing CBM pools

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): The number of stands, and therefore rows in the
            resulting dataframe.
        pool_codes (list): a list of pool names, which are used as column
            labels in the resulting dataframe

    Returns:
        pandas.DataFrame: A dataframe for storing CBM pools
    """
    pools = pd.DataFrame(
        data=np.zeros((n_stands, len(pool_codes))),
        columns=pool_codes)

    # By convention the libcbm CBM implementation uses an input pool at
    # index 0 whose value is always 1.0.
    # TODO: move this into the lower level code, since it is a model behaviour
    pools[pool_codes[0]] = 1.0

    return pools


def initialize_flux(n_stands, flux_indicator_codes):
    """Create a dataframe for storing CBM flux indicator values

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): The number of stands, and therefore rows in the
            resulting dataframe.
        flux_indicator_codes (list): a list of flux indicator names, which
            are used as column labels in the resulting dataframe

    Returns:
        pandas.DataFrame: A dataframe for storing CBM flux indicators
    """
    return pd.DataFrame(
        data=np.zeros((n_stands, len(flux_indicator_codes))),
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

    If a scalar value is provided to any of the optional parameters, that
    value will be promoted (see: `promote_scalar`) in the resulting vector.

    Args:
        n_stands (int): The length of each of the resulting variables
            vectors returned by this function.
        return_interval (numpy.ndarray or number, optional): The number of
            years between historical disturbances in the spinup function.
            Defaults to None.
        min_rotations (numpy.ndarray or number, optional): The minimum number
            of historical rotations to perform. Defaults to None.
        max_rotations (numpy.ndarray or number, optional): The maximum number
            of historical rotations to perform. Defaults to None.
        mean_annual_temp (numpy.ndarray or number, optional): The mean annual
            temperature used in the spinup procedure. Defaults to None.

    Returns:
        object: Returns an object with properties to access each of the
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


def initialize_spinup_variables(n_stands):
    """Creates a collection of vectors used as working/state variables for
    the spinup routine.

    Args:
        n_stands (int): The number of stands

    Returns:
        object: an object with properties to access working variables
            needed by the spinup routine.
    """

    variables = SimpleNamespace()
    variables.spinup_state = np.zeros(n_stands, dtype=np.uint32)
    variables.slow_pools = np.zeros(n_stands, dtype=np.float)
    variables.disturbance_type = np.zeros(n_stands, dtype=np.int32)
    variables.rotation = np.zeros(n_stands, dtype=np.int32)
    variables.step = np.zeros(n_stands, dtype=np.int32)
    variables.last_rotation_slow_C = np.zeros(n_stands, dtype=np.float)
    variables.enabled = np.ones(n_stands, dtype=np.int32)
    variables.age = np.zeros(n_stands, dtype=np.int32)
    variables.growth_enabled = np.ones(n_stands, dtype=np.int32)

    # these variables are not used during spinup, but are needed
    # for CBM function signatures, and will be passed as nulls
    variables.last_disturbance_type = None
    variables.time_since_last_disturbance = None
    variables.growth_multiplier = None
    return variables


def initialize_cbm_parameters(n_stands, disturbance_type=0,
                              transition_id=0, mean_annual_temp=None):
    """Create CBM parameters as a collection of variable vectors

    The variables here are all of length N stands and are row-aligned with
    all other vectors and dataframes using this convention.

    The mean_annual temperature keyword argument is optional, and if
    unspecified, libcbm will use a default for the corresponding parameter
    drawn from the cbm_defaults database.

    If a scalar value is provided to any of the optional parameters, that value
    will be filled in the resulting vector.

    Args:
        n_stands (int): The number of stands
        disturbance_type (numpy.ndarray or int, optional): The disturbance
            type id which references the disturbance types defined in the
            libCBM configuration.  By convention, a negative or 0 value
            indicates no disturbance. Defaults to 0.
        transition_id (numpy.ndarray, int, optional): The transition id which
            references the transition rules defined in the libCBM
            configuration.  By convention, a negative or 0 value indicates no
            transition. Defaults to 0.
        mean_annual_temp (numpy.ndarray, int, optional): A value, in degrees
            Celsius, that defines this timestep's mean annual temperature for
            each stand. Defaults to None.

    Returns:
        object: an object with properties for each cbm parameter used by
            the cbm step function.
    """
    parameters = SimpleNamespace()
    parameters.disturbance_type = promote_scalar(
        disturbance_type, n_stands, dtype=np.int32)
    parameters.transition_rule_id = promote_scalar(
        transition_id, n_stands, dtype=np.int32)
    parameters.mean_annual_temp = promote_scalar(
        mean_annual_temp, n_stands, dtype=np.float)
    return parameters


def initialize_cbm_state_variables(n_stands):
    """Creates a pandas dataframe containing state variables used by CBM
    functions at simulation runtime, with default initial values.

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): the number of rows in the resulting dataframe.

    Returns:
        pandas.DataFrame: a dataframe containing the CBM state variables.
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


def initialize_inventory(classifiers, inventory):
    """Creates input for `libcbm.model.cbm.CBM` functions

    Args:
        classifiers (pandas.DataFrame): dataframe of inventory classifier
            sets. Column names are the name of the classifiers, and values
            are the ids for each classifier value associated with the
            inventory at each row.
        inventory (pandas.DataFrame): Data defining the inventory. Columns:

            - age: the inventory age at the start of CBM simulation
            - spatial_unit: the spatial unit id
            - afforestation_pre_type_id: If the stand is initially
                non-forested, this can be used to set an initial soil
                condition.
            - land_class: a UNFCCC land class code.
            - historical_disturbance_type: the id for a disturbance type
                used for the historical disturbance rotations in the spinup
                routine.
            - last_pass_disturbance_type: the id for a disturbance type used
              for the final disturbance in the spinup routine.
    Raises:
        ValueError: Raised if the number of rows for classifiers and
            data are not the same.

    Returns:
        object: an object containing the inventory and classifier
            data.
    """
    n_stands = len(inventory.index)
    if not len(classifiers.index) == n_stands:
        raise ValueError(
            ("number of inventory records: {inv} does not match number of "
             "classifier sets: {c_sets}").format(
                inv=n_stands, c_sets=len(classifiers.index)))
    i = SimpleNamespace()
    i.classifiers = classifiers
    i.age = inventory.age.to_numpy()
    i.spatial_unit = inventory.spatial_unit.to_numpy()
    i.afforestation_pre_type_id = \
        inventory.afforestation_pre_type_id.to_numpy()
    i.land_class = inventory.land_class.to_numpy()
    i.historical_disturbance_type = \
        inventory.historic_disturbance_type.to_numpy()
    i.last_pass_disturbance_type = \
        inventory.last_pass_disturbance_type.to_numpy()
    i.delay = inventory.delay.to_numpy()
    return i
