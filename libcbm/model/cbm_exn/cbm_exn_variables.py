from __future__ import annotations
import numpy as np
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.storage.backends import BackendType
from libcbm.storage.series import SeriesDef
from libcbm.model.model_definition.spinup_engine import SpinupState


def init_pools(
    n_rows: int, pool_names: list[str], backend_type: BackendType
) -> DataFrame:
    """Initialize the pools dataframe for cbm_vars. The values are all set to
    zero.

    Args:
        n_rows (int): the number of rows in the resulting dataframe
        pool_names (list[str]): the list of pool names, this forms the columns
            of the resulting dataframe.
        backend_type (BackendType): The backend storage type

    Returns:
        DataFrame: initialized dataframe
    """
    return dataframe.numeric_dataframe(pool_names, n_rows, backend_type)


def init_flux(
    n_rows: int, flux_names: list[str], backend_type: BackendType
) -> DataFrame:
    """Initialize the flux dataframe for cbm_vars. The values are all set to
    zero.

    Args:
        n_rows (int): the number of rows in the resulting dataframe
        flux_names (list[str]): the list of flux names, this forms the columns
            of the resulting dataframe.
        backend_type (BackendType): The backend storage type

    Returns:
        DataFrame: initialized dataframe
    """
    return dataframe.numeric_dataframe(flux_names, n_rows, backend_type)


def init_parameters(n_rows: int, backend_type: BackendType) -> DataFrame:
    """Initialize the flux dataframe for cbm_vars. The values are all set to
    zero in the case of integer columns, and np.nan for float columns

    Columns::

        * mean_annual_temperature (float) - the mean annual tempurature
            step parameter
        * disturbance_type (int) - the disturbance type id step parameter
        * merch_inc (float) - net aboveground merchantable C increment step
            parameter
        * foliage_inc (float) - net aboveground foliage C increment step
            parameter
        * other_inc (float) - net aboveground other C increment step
            parameter


    Args:
        n_rows (int): the number of rows in the resulting dataframe
        backend_type (BackendType): The backend storage type

    Returns:
        DataFrame: initialized dataframe
    """
    return dataframe.from_series_list(
        [
            SeriesDef("mean_annual_temperature", np.nan, "float"),
            SeriesDef("disturbance_type", 0, "int32"),
            SeriesDef("merch_inc", np.nan, "float"),
            SeriesDef("foliage_inc", np.nan, "float"),
            SeriesDef("other_inc", np.nan, "float"),
        ],
        nrows=n_rows,
        back_end=backend_type,
    )


def init_state(n_rows: int, backend_type: BackendType) -> DataFrame:
    """Intitialze a dataframe containing CBM state variables. The values are
    all set to zero in the case of integer columns, and np.nan for float
    columns.

    Identifiers correspond to values in the default parameters.
    See :py:func:`libcbm.model.cbm_exn.cbm_exn_parameters`

    Columns::

        * area (float) - area in hectares
        * spatial_unit_id (int) - spatial unit identifier
        * land_class_id (int) - UNFCCC land class identifier
        * age (int) - the age
        * species (int) - species identifier
        * sw_hw (int) - flag indicating softwood when 0 and hardwood when 1
        * time_since_last_disturbance (int) - the number of timesteps since
            the last disturbance event
        * time_since_land_use_change (int) - the number of timesteps since a
            UNFCCC land class change occurred, or -1 if no change has occurred
        * last_disturbance_type (int) - the id of the last disturbance type to
            occur
        * enabled (int) - flag indicating all processes are disabled when 1
            and otherwise fully disabled meaning no changes to cbm_vars should
            occur

    Args:
        n_rows (int): the number of rows in the resulting dataframe
        backend_type (BackendType): The backend storage type

    Returns:
        DataFrame: initialized dataframe
    """
    return dataframe.from_series_list(
        [
            SeriesDef("area", np.nan, "float"),
            SeriesDef("spatial_unit_id", 0, "int32"),
            SeriesDef("land_class_id", 0, "int32"),
            SeriesDef("age", 0, "int32"),
            SeriesDef("species", 0, "int32"),
            SeriesDef("sw_hw", 0, "int32"),
            SeriesDef("time_since_last_disturbance", 0, "int32"),
            SeriesDef("time_since_land_use_change", 0, "int32"),
            SeriesDef("last_disturbance_type", 0, "int32"),
            SeriesDef("enabled", 0, "int32"),
        ],
        nrows=n_rows,
        back_end=backend_type,
    )


def init_spinup_state(n_rows: int, backend_type: BackendType) -> DataFrame:
    """Intitialze a dataframe containing CBM state variables specific to the
    spinup process. The values are all set to zero in the case of integer
    columns, and np.nan for float columns.

    Columns::

        * spinup_state - see:
            :py:class:`libcbm.model.model_definition.spinup_engine.SpinupState`
        * age - the age
        * delay_step - tracks the number of steps when in the delay state
        * disturbance_type - set to indicate a historical or last pass
            disturbance should occur.
        * rotation - the number of rotations that have occurred
        * last_rotation_slow - the sum of slow C values at the end of the last
            rotation
        * this_rotation_slow - the sum of slow C values for this rotation
        * enabled - set to 0 when spinup has finished for the corresponding
            dataframe row

    Args:
        n_rows (int): the number of rows in the resulting dataframe
        backend_type (BackendType): The backend storage type

    Returns:
        DataFrame: initialized dataframe
    """
    return dataframe.from_series_list(
        [
            SeriesDef("spinup_state", SpinupState.AnnualProcesses, "int"),
            SeriesDef("age", 0, "int"),
            SeriesDef("delay_step", 0, "int"),
            SeriesDef("disturbance_type", 0, "int"),
            SeriesDef("rotation", 0, "int"),
            SeriesDef("last_rotation_slow", 0, "float"),
            SeriesDef("this_rotation_slow", 0, "float"),
            SeriesDef("enabled", 1, "int"),
        ],
        nrows=n_rows,
        back_end=backend_type,
    )


def init_cbm_vars(
    n_rows: int,
    pool_names: list[str],
    flux_names: list[str],
    backend_type: BackendType,
) -> CBMVariables:
    """Initialize dataframe storage for cbm variables and state.

    Member dataframes of cbm_vars::

        * pools: the CBM pools
        * flux: dataframe for tracking flux indicator values (akin to CBM-CFS3
            tblFluxIndicators)
        * parameters: timestep parameters for each stand
        * state: state variables

    Args:
        n_rows (int): The number of rows in each member dataframe.
        pool_names (list[str]): The list of pool names, defines the columns
            of the cbm_vars `pools` dataframe.
        flux_names (list[str]): The list of flux indicator names, defines the
            columns of the cbm_vars `flux` dataframe.
        backend_type (BackendType): The backend storage type

    Returns:
        CBMVariables: Initialized cbm_vars
    """

    return CBMVariables(
        {
            "pools": init_pools(n_rows, pool_names, backend_type),
            "flux": init_flux(n_rows, flux_names, backend_type),
            "parameters": init_parameters(n_rows, backend_type),
            "state": init_state(n_rows, backend_type),
        }
    )
