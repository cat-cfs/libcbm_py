# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.dataframe import Series
from libcbm.storage.dataframe import NullSeries
from libcbm.storage.backends import BackendType


class CBMVariables:
    def __init__(self):
        pass

    @property
    def pools(self) -> DataFrame:
        pass

    @property
    def flux(self) -> DataFrame:
        pass

    @property
    def classifiers(self) -> DataFrame:
        pass

    @property
    def state(self) -> DataFrame:
        pass

    @property
    def inventory(self) -> DataFrame:
        pass

    @property
    def parameters(self) -> DataFrame:
        pass


def _initialize_pools(
    n_stands: int, pool_codes: list[str], back_end: BackendType
) -> DataFrame:
    """Create a dataframe for storing CBM pools

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): The number of stands, and therefore rows in the
            resulting dataframe.
        pool_codes (list): a list of pool names, which are used as column
            labels in the resulting dataframe
        back_end (BackendType): the storage type for the pools

    Returns:
        DataFrame: A dataframe for storing CBM pools
    """
    pools = dataframe.numeric_dataframe(
        cols=pool_codes,
        nrows=n_stands,
        back_end=back_end,
    )

    # By convention the libcbm CBM implementation uses an input pool at
    # index 0 whose value is always 1.0.
    # TODO: move this into the lower level code, since it is a model behaviour
    pools[pool_codes[0]] = 1.0

    return pools


def _initialize_flux(
    n_stands: int, flux_indicator_codes: list[str], back_end: BackendType
) -> DataFrame:
    """Create a dataframe for storing CBM flux indicator values

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): The number of stands, and therefore rows in the
            resulting dataframe.
        flux_indicator_codes (list): a list of flux indicator names, which
            are used as column labels in the resulting dataframe
        back_end (BackendType): the storage type for the pools

    Returns:
        DataFrame: A dataframe for storing CBM flux indicators
    """
    return dataframe.numeric_dataframe(
        cols=flux_indicator_codes,
        nrows=n_stands,
        back_end=back_end,
    )


def initialize_spinup_parameters(
    n_stands: int,
    back_end: BackendType = BackendType.numpy,
    return_interval: dataframe.SeriesInitType = None,
    min_rotations: dataframe.SeriesInitType = None,
    max_rotations: dataframe.SeriesInitType = None,
    mean_annual_temp: dataframe.SeriesInitType = None,
) -> DataFrame:
    """Create spinup parameters as a collection of variable vectors

    The variables here are all of length N stands and are row-aligned with
    all other vectors and dataframes using this convention.

    Each keyword argument is optional, and if unspecified, libcbm will use a
    default for the corresponding parameter drawn from cbm_defaults.  These
    parameters are available here to override those default values on a
    per-stand basis.

    If a scalar value is provided to any of the optional parameters, that
    value will be promoted to a Series in the resulting
    vector.

    Args:
        n_stands (int): The length of each of the resulting variables
            vectors returned by this function.
        return_interval (SeriesInitType, optional): The number of
            years between historical disturbances in the spinup function.
            Defaults to None.
        min_rotations (SeriesInitType, optional): The minimum number
            of historical rotations to perform. Defaults to None.
        max_rotations (SeriesInitType, optional): The maximum number
            of historical rotations to perform. Defaults to None.
        mean_annual_temp (SeriesInitType, optional): The mean annual
            temperature used in the spinup procedure. Defaults to None.

    Returns:
        DataFrame: table of spinup paramaeters
    """

    def make_series(name, init, type):
        if init is None:
            return NullSeries(name)
        else:
            return Series(name, init, type)

    data = [
        make_series("return_interval", return_interval, "int32"),
        make_series("min_rotations", min_rotations, "int32"),
        make_series("max_rotations", max_rotations, "int32"),
        make_series("mean_annual_temp", mean_annual_temp, "float64"),
    ]
    parameters = DataFrame(data, n_stands, back_end)

    return parameters


def _initialize_spinup_state_variables(
    n_stands: int, back_end: BackendType
) -> DataFrame:
    """Creates a collection of vectors used as working/state variables for
    the spinup routine.

    Args:
        n_stands (int): The number of stands

    Returns:
        DataFrame: table of working variables
            needed by the spinup routine.
    """
    # favouring SimpleNamespace over pd.DataFrame here because these are
    # null variables, and DataFrame does not support null columns

    variables = DataFrame(
        data=[
            Series("spinup_state", 0, n_stands, "uint32"),
            Series("slow_pools", 0, n_stands, "float64"),
            Series("disturbance_type", 0, n_stands, "int32"),
            Series("rotation", 0, n_stands, "int32"),
            Series("step", 0, n_stands, "int32"),
            Series("last_rotation_slow_C", 0, n_stands, "float64"),
            Series("enabled", 0, n_stands, "int32"),
            Series("age", 0, n_stands, "np.int32"),
            Series("growth_enabled", 0, n_stands, "int32"),
            # these variables are not used during spinup, but are needed
            # for CBM function signatures, and will be passed as nulls
            NullSeries("last_disturbance_type"),
            NullSeries("time_since_last_disturbance"),
            NullSeries("growth_multiplier"),
        ],
        nrows=n_stands,
        back_end=back_end,
    )

    return variables


def _initialize_cbm_parameters(
    n_stands: int,
    back_end: BackendType,
    disturbance_type: dataframe.SeriesInitType = 0,
    reset_age: dataframe.SeriesInitType = -1,
    mean_annual_temp: dataframe.SeriesInitType = None,
) -> DataFrame:
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
        disturbance_type (SeriesInitType, optional): The disturbance
            type id which references the disturbance types defined in the
            libCBM configuration.  By convention, a negative or 0 value
            indicates no disturbance. Defaults to 0.
        reset_age (SeriesInitType, int, optional): The post disturbance reset
            age. By convention, a negative value indicates a null reset_age.
            If the null reset_age value is specified the post disturbance age
            will be determined by the following decision:

                - if the disturbance type results in no biomass after the
                  disturbance matrix is applied, the age is reset to zero.
                - otherwise, the age is not modified from the pre-disturbance
                  age.

            Defaults to -1
        mean_annual_temp (SeriesInitType, optional): A value, in degrees
            Celsius, that defines this timestep's mean annual temperature for
            each stand. Defaults to None.

    Returns:
        DataFrame: dataframe with CBM timestep parameters as columns
            and 1 row per stand.
    """

    data = [
        Series("disturbance_type", disturbance_type, "int32"),
        Series("reset_age", reset_age, "int32"),
    ]
    if mean_annual_temp:
        data.append(Series("mean_annual_temp", mean_annual_temp, "float64"))
    else:
        data.append(NullSeries("mean_annual_temp"))
    parameters = DataFrame(data, n_stands, back_end)
    return parameters


def _initialize_cbm_state_variables(
    n_stands: int, back_end: BackendType
) -> DataFrame:
    """Creates a dataframe containing state variables used by CBM
    functions at simulation runtime, with default initial values.

    The dataframe here has 1 row for each stand and is row-aligned with
    all other vectors and dataframes using this convention.

    Args:
        n_stands (int): the number of rows in the resulting dataframe.

    Returns:
        DataFrame: a dataframe containing the CBM state variables.
    """
    data = [
        Series("last_disturbance_type", 0, "int32"),
        Series("time_since_last_disturbance", 0, "int32"),
        Series("time_since_land_class_change", -1, "int32"),
        Series("growth_enabled", 0, "int32"),
        Series("enabled", 1, "int32"),
        Series("land_class", 0, "int32"),
        Series("age", 0, "int32"),
        Series("growth_multiplier", 1.0, "float64"),
        Series("regeneration_delay", 0, "int32"),
    ]
    state_variables = DataFrame(data, n_stands, back_end)

    return state_variables


def _initialize_inventory(
    inventory: DataFrame, back_end: BackendType
) -> DataFrame:
    """Check fields and types of inventory input for
    :class:`libcbm.model.cbm.cbm_model.CBM` functions

    Converts inventory to specified backend

    Example Inventory table: (abbreviated column names)

    ====  ==== =====  ======  =====  ====  =====  =====
     age  area  spu   affor    lc    hist  last   delay
    ====  ==== =====  ======  =====  ====  =====  =====
     0     5      1      1      1     1      3     10
     11    7     10     -1      0     1      1     -1
     0    30     42      1      0     1      5     -1
    ====  ==== =====  ======  =====  ====  =====  =====

    Args:

        inventory (DataFrame): Data defining the inventory. Columns:

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
        DataFrame: dataframe containing the inventory data.
    """

    data = [
        Series("age", inventory.age, "int32"),
        Series("area", inventory.area, "float64"),
        Series("spatial_unit", inventory.spatial_unit, "int32"),
        Series(
            "afforestation_pre_type_id",
            inventory.afforestation_pre_type_id,
            "int32",
        ),
        Series("land_class", inventory.land_class, "int32"),
        Series(
            "historical_disturbance_type",
            inventory.historical_disturbance_type,
            "int32",
        ),
        Series(
            "last_pass_disturbance_type",
            inventory.last_pass_disturbance_type,
            "int32",
        ),
        Series("delay", inventory.delay, "int32"),
    ]
    return DataFrame(data, inventory.n_rows, back_end)


def _initialize_classifiers(
    classifiers: DataFrame, back_end: BackendType
) -> DataFrame:
    """converts classifiers table to required type"""
    data = [
        Series(name, classifiers[name], "int32")
        for name in classifiers.columns
    ]
    return DataFrame(data, classifiers.n_rows, back_end)


def initialize_spinup_variables(
    cbm_vars: CBMVariables,
    spinup_params: DataFrame = None,
    include_flux: bool = False,
) -> CBMVariables:

    n_stands = cbm_vars.inventory.shape[0]
    if spinup_params is None:
        spinup_params = initialize_spinup_parameters(n_stands)

    spinup_vars = CBMVariables()
    spinup_vars.pools = cbm_vars.pools
    spinup_vars.flux = cbm_vars.flux if include_flux else None
    spinup_vars.parameters = spinup_params
    spinup_vars.state = _initialize_spinup_state_variables(n_stands)
    spinup_vars.inventory = cbm_vars.inventory
    spinup_vars.classifiers = cbm_vars.classifiers
    return spinup_vars


def initialize_simulation_variables(
    classifiers: DataFrame,
    inventory: DataFrame,
    pool_codes: list[str],
    flux_indicator_codes: list[str],
) -> CBMVariables:
    """Packages and initializes the cbm variables (cbm_vars) as an object with
    named properties

    Args:
        classifiers (DataFrame): DataFrame of integer classifier value
            ids.  Rows are stands and cols are classifiers
        inventory (DataFrame): The inventory to simulate. Each row
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
    cbm_vars = CBMVariables()
    cbm_vars.pools = _initialize_pools(n_stands, pool_codes)
    cbm_vars.flux = _initialize_flux(n_stands, flux_indicator_codes)
    cbm_vars.parameters = _initialize_cbm_parameters(n_stands)
    cbm_vars.state = _initialize_cbm_state_variables(n_stands)
    cbm_vars.inventory = _initialize_inventory(inventory)
    cbm_vars.classifiers = _initialize_classifiers(classifiers)
    return cbm_vars
