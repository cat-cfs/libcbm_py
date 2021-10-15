# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import pandas as pd
import numpy as np
from types import SimpleNamespace
from libcbm import data_helpers


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
    value will be promoted (see: :py:func:`promote_scalar`) in the resulting
    vector.

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

    # favouring SimpleNamespace over pd.DataFrame here because these are
    # potentially null variables, and DataFrame does not support null columns

    parameters = SimpleNamespace()
    parameters.return_interval = data_helpers.promote_scalar(
        return_interval, n_stands, dtype=np.int32)
    parameters.min_rotations = data_helpers.promote_scalar(
        min_rotations, n_stands, dtype=np.int32)
    parameters.max_rotations = data_helpers.promote_scalar(
        max_rotations, n_stands, dtype=np.int32)
    parameters.mean_annual_temp = data_helpers.promote_scalar(
        mean_annual_temp, n_stands, dtype=np.float64)
    return parameters


def initialize_spinup_state_variables(n_stands):
    """Creates a collection of vectors used as working/state variables for
    the spinup routine.

    Args:
        n_stands (int): The number of stands

    Returns:
        object: an object with properties to access working variables
            needed by the spinup routine.
    """
    # favouring SimpleNamespace over pd.DataFrame here because these are
    # null variables, and DataFrame does not support null columns

    variables = SimpleNamespace()
    variables.spinup_state = np.zeros(n_stands, dtype=np.uint32)
    variables.slow_pools = np.zeros(n_stands, dtype=np.float64)
    variables.disturbance_type = np.zeros(n_stands, dtype=np.int32)
    variables.rotation = np.zeros(n_stands, dtype=np.int32)
    variables.step = np.zeros(n_stands, dtype=np.int32)
    variables.last_rotation_slow_C = np.zeros(n_stands, dtype=np.float64)
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
                              reset_age=-1, mean_annual_temp=None):
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
        reset_age (numpy.ndarray, int, optional): The post disturbance reset
            age. By convention, a negative value indicates a null reset_age.
            If the null reset_age value is specified the post disturbance age
            will be determined by the following decision:

                - if the disturbance type results in no biomass after the
                  disturbance matrix is applied, the age is reset to zero.
                - otherwise, the age is not modified from the pre-disturbance
                  age.

            Defaults to -1
        mean_annual_temp (numpy.ndarray, float, optional): A value, in degrees
            Celsius, that defines this timestep's mean annual temperature for
            each stand. Defaults to None.

    Returns:
        pandas.DataFrame: dataframe with CBM timestep parameters as columns
            and 1 row per stand.
    """

    data = {
        "disturbance_type": data_helpers.promote_scalar(
            disturbance_type, n_stands, dtype=np.int32),
        "reset_age": data_helpers.promote_scalar(
            reset_age, n_stands, dtype=np.int32)
    }
    if mean_annual_temp:
        data["mean_annual_temp"] = data_helpers.promote_scalar(
            mean_annual_temp, n_stands, dtype=np.float64)
    parameters = pd.DataFrame(data=data)
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
    state_variables = pd.DataFrame({
        "last_disturbance_type": np.zeros(n_stands, dtype=np.int32),
        "time_since_last_disturbance": np.zeros(n_stands, dtype=np.int32),
        "time_since_land_class_change": np.ones(n_stands, dtype=np.int32) * -1,
        "growth_enabled": np.zeros(n_stands, dtype=np.int32),
        "enabled": np.ones(n_stands, dtype=np.int32),
        "land_class": np.zeros(n_stands, dtype=np.int32),
        "age": np.zeros(n_stands, dtype=np.int32),
        "growth_multiplier": np.ones(n_stands, dtype=np.float64),
        "regeneration_delay": np.zeros(n_stands, dtype=np.int32)
    })

    return state_variables


def initialize_inventory(inventory):
    """Check fields and types of inventory input for
    :class:`libcbm.model.cbm.cbm_model.CBM` functions

    Example Inventory table: (abbreviated column names)

    ====  ==== =====  ======  =====  ====  =====  =====
     age  area  spu   affor    lc    hist  last   delay
    ====  ==== =====  ======  =====  ====  =====  =====
     0     5      1      1      1     1      3     10
     11    7     10     -1      0     1      1     -1
     0    30     42      1      0     1      5     -1
    ====  ==== =====  ======  =====  ====  =====  =====

    Args:

        inventory (pandas.DataFrame): Data defining the inventory. Columns:

            - age: the inventory age at the start of CBM simulation
            - area: the inventory area
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
            - delay: number of steps to simulate dead organic matter decays
              after a last pass deforestation event occurs.

    Returns:
        pandas.DataFrame: dataframe containing the inventory data.
    """
    return pd.DataFrame({
        "age": inventory.age.to_numpy(dtype=np.int32),
        "area": inventory.area.to_numpy(dtype=np.float64),
        "spatial_unit": inventory.spatial_unit.to_numpy(dtype=np.int32),
        "afforestation_pre_type_id":
            inventory.afforestation_pre_type_id.to_numpy(dtype=np.int32),
        "land_class": inventory.land_class.to_numpy(dtype=np.int32),
        "historical_disturbance_type":
            inventory.historical_disturbance_type.to_numpy(dtype=np.int32),
        "last_pass_disturbance_type":
            inventory.last_pass_disturbance_type.to_numpy(dtype=np.int32),
        "delay": inventory.delay.to_numpy(dtype=np.int32)})


def initialize_classifiers(classifiers):
    """converts classifiers table to required type"""
    return pd.DataFrame(
        data=classifiers.to_numpy(dtype=np.int32),
        columns=list(classifiers))


def _make_contiguous(df):
    """Orders the underlying memory in a numpy-backed dataframe as C contiguous
    (row major ordering)

    Args:
        df (pandas.DataFrame): a pandas dataframe

    Returns:
        pandas.DataFrame: a C contiguous copy of the input data frame.
    """

    if not df.values.flags["C_CONTIGUOUS"]:
        return pd.DataFrame(
            columns=df.columns.tolist(),
            data=np.ascontiguousarray(df))
    return df


def prepare(cbm_vars):
    """prepares, validates the specified cbm_vars object for use with low
    level functions

    Args:
        cbm_vars (object): the cbm variables to validate and prepare
    """

    for field in ["pools", "flux", "classifiers"]:
        if field in cbm_vars.__dict__:
            cbm_vars.__dict__[field] = \
                _make_contiguous(cbm_vars.__dict__[field])

    return cbm_vars


def initialize_spinup_variables(cbm_vars, spinup_params=None,
                                include_flux=False):

    n_stands = cbm_vars.inventory.shape[0]
    if spinup_params is None:
        spinup_params = initialize_spinup_parameters(n_stands)

    spinup_vars = SimpleNamespace()
    spinup_vars.pools = cbm_vars.pools
    spinup_vars.flux = cbm_vars.flux if include_flux else None
    spinup_vars.parameters = spinup_params
    spinup_vars.state = initialize_spinup_state_variables(n_stands)
    spinup_vars.inventory = cbm_vars.inventory
    spinup_vars.classifiers = cbm_vars.classifiers
    return spinup_vars


def initialize_simulation_variables(classifiers, inventory, pool_codes,
                                    flux_indicator_codes):
    """Packages and initializes the cbm variables (cbm_vars) as an object with
    named properties

    Args:
        classifiers (pandas.DataFrame): DataFrame of integer classifier value
            ids.  Rows are stands and cols are classifiers
        inventory (pandas.DataFrame): The inventory to simulate. Each row
            represents a stand. See :py:func:`initialize_inventory` for a
            description of the required columns.
        pool_codes (list): the list of string pool names which act as column
            labels for the resulting cbm_vars.pools DataFrame.
        flux_indicator_codes (list): the list of string flux indicator names
            which act as column labels for the resulting
            cbm_vars.flux DataFrame.

    Returns:
        object: Returns the cbm_vars object for simulating CBM.
    """
    n_stands = inventory.shape[0]
    cbm_vars = SimpleNamespace()
    cbm_vars.pools = initialize_pools(n_stands, pool_codes)
    cbm_vars.flux = initialize_flux(n_stands, flux_indicator_codes)
    cbm_vars.parameters = initialize_cbm_parameters(n_stands)
    cbm_vars.state = initialize_cbm_state_variables(n_stands)
    cbm_vars.inventory = initialize_inventory(inventory)
    cbm_vars.classifiers = initialize_classifiers(classifiers)
    return prepare(cbm_vars)
