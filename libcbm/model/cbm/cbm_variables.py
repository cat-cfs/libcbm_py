# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import SeriesDef
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType


class CBMVariables:
    def __init__(
        self,
        pools: DataFrame,
        flux: DataFrame,
        classifiers: DataFrame,
        state: DataFrame,
        inventory: DataFrame,
        parameters: DataFrame,
    ):
        self._pools = pools
        self._flux = flux
        self._classifiers = classifiers
        self._state = state
        self._inventory = inventory
        self._parameters = parameters

    @property
    def pools(self) -> DataFrame:
        return self._pools

    @pools.setter
    def pools(self, df: DataFrame):
        self._pools = dataframe.convert_dataframe_backend(
            df, self._pools.backend_type
        )

    @property
    def flux(self) -> DataFrame:
        return self._flux

    @flux.setter
    def flux(self, df: DataFrame):
        self._flux = dataframe.convert_dataframe_backend(
            df, self._flux.backend_type
        )

    @property
    def classifiers(self) -> DataFrame:
        return self._classifiers

    @classifiers.setter
    def classifiers(self, df: DataFrame):
        self._classifiers = dataframe.convert_dataframe_backend(
            df, self._classifiers.backend_type
        )

    @property
    def state(self) -> DataFrame:
        return self._state

    @state.setter
    def state(self, df: DataFrame):
        self._state = dataframe.convert_dataframe_backend(
            df, self._state.backend_type
        )

    @property
    def inventory(self) -> DataFrame:
        return self._inventory

    @inventory.setter
    def inventory(self, df: DataFrame):
        self._inventory = dataframe.convert_dataframe_backend(
            df, self._inventory.backend_type
        )

    @property
    def parameters(self) -> DataFrame:
        return self._parameters

    @parameters.setter
    def parameters(self, df: DataFrame):
        self._parameters = dataframe.convert_dataframe_backend(
            df, self._parameters.backend_type
        )


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
    pools[pool_codes[0]].assign(1.0)

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
    return_interval: Series = None,
    min_rotations: Series = None,
    max_rotations: Series = None,
    mean_annual_temp: Series = None,
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
        return_interval (Series, optional): The number of
            years between historical disturbances in the spinup function.
            Defaults to None.
        min_rotations (Series, optional): The minimum number
            of historical rotations to perform. Defaults to None.
        max_rotations (Series, optional): The maximum number
            of historical rotations to perform. Defaults to None.
        mean_annual_temp (Series, optional): The mean annual
            temperature used in the spinup procedure. Defaults to None.

    Returns:
        DataFrame: table of spinup paramaeters
    """
    data = []
    for s in [return_interval, min_rotations, max_rotations, mean_annual_temp]:
        if s is not None:
            data.append(s)
    parameters = dataframe.from_series_list(
        data,
        nrows=n_stands,
        back_end=back_end,
    )

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

    variables = dataframe.from_series_list(
        [
            SeriesDef("spinup_state", 0, "uint32"),
            SeriesDef("slow_pools", 0, "float64"),
            SeriesDef("disturbance_type", 0, "int32"),
            SeriesDef("rotation", 0, "int32"),
            SeriesDef("step", 0, "int32"),
            SeriesDef("last_rotation_slow_C", 0, "float64"),
            SeriesDef("enabled", 1, "int32"),
            SeriesDef("age", 0, "int32"),
            SeriesDef("growth_enabled", 1, "int32"),
        ],
        nrows=n_stands,
        back_end=back_end,
    )

    return variables


def _initialize_cbm_parameters(
    n_stands: int,
    back_end: BackendType,
) -> DataFrame:
    """Create CBM parameters

    Args:
        n_stands (int): The number of stands

    Returns:
        DataFrame: dataframe with CBM timestep parameters as columns
            and 1 row per stand.
    """

    data = [
        SeriesDef("disturbance_type", 0, "int32"),
        SeriesDef("reset_age", -1, "int32"),
    ]

    parameters = dataframe.from_series_list(data, n_stands, back_end)
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
        SeriesDef("last_disturbance_type", 0, "int32"),
        SeriesDef("time_since_last_disturbance", 0, "int32"),
        SeriesDef("time_since_land_class_change", -1, "int32"),
        SeriesDef("growth_enabled", 0, "int32"),
        SeriesDef("enabled", 1, "int32"),
        SeriesDef("land_class", 0, "int32"),
        SeriesDef("age", 0, "int32"),
        SeriesDef("growth_multiplier", 1.0, "float64"),
        SeriesDef("regeneration_delay", 0, "int32"),
    ]
    state_variables = dataframe.from_series_list(data, n_stands, back_end)

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
        inventory["age"].as_type("int32"),
        inventory["area"].as_type("float64"),
        inventory["spatial_unit"].as_type("int32"),
        inventory["afforestation_pre_type_id"].as_type("int32"),
        inventory["land_class"].as_type("int32"),
        inventory["historical_disturbance_type"].as_type("int32"),
        inventory["last_pass_disturbance_type"].as_type("int32"),
        inventory["delay"].as_type("int32"),
    ]
    if "spatial_reference" in inventory.columns:
        data.append(inventory["spatial_reference"])
    return dataframe.from_series_list(data, inventory.n_rows, back_end)


def _initialize_classifiers(
    classifiers: DataFrame, back_end: BackendType
) -> DataFrame:
    """converts classifiers table to required type"""
    data = [classifiers[name].as_type("int32") for name in classifiers.columns]
    return dataframe.from_series_list(data, classifiers.n_rows, back_end)


def initialize_spinup_variables(
    cbm_vars: CBMVariables,
    backend_type: BackendType,
    spinup_params: DataFrame = None,
    include_flux: bool = False,
) -> CBMVariables:

    n_stands = cbm_vars.inventory.n_rows
    if spinup_params is None:
        spinup_params = initialize_spinup_parameters(n_stands, backend_type)

    spinup_vars = CBMVariables(
        cbm_vars.pools,
        cbm_vars.flux if include_flux else None,
        cbm_vars.classifiers,
        _initialize_spinup_state_variables(n_stands, backend_type),
        cbm_vars.inventory,
        spinup_params,
    )

    return spinup_vars


def initialize_simulation_variables(
    classifiers: DataFrame,
    inventory: DataFrame,
    pool_codes: list[str],
    flux_indicator_codes: list[str],
    backend_type: BackendType,
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
    n_stands = inventory.n_rows
    cbm_vars = CBMVariables(
        _initialize_pools(n_stands, pool_codes, backend_type),
        _initialize_flux(n_stands, flux_indicator_codes, backend_type),
        _initialize_classifiers(classifiers, backend_type),
        _initialize_cbm_state_variables(n_stands, backend_type),
        _initialize_inventory(inventory, backend_type),
        _initialize_cbm_parameters(n_stands, backend_type),
    )

    return cbm_vars
