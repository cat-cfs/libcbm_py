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
    return dataframe.numeric_dataframe(pool_names, n_rows, backend_type)


def init_flux(
    n_rows: int, flux_names: list[str], backend_type: BackendType
) -> DataFrame:
    return dataframe.numeric_dataframe(flux_names, n_rows, backend_type)


def init_parameters(n_rows: int, backend_type: BackendType):
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


def init_state(n_rows: int, backend_type: BackendType):
    return dataframe.from_series_list(
        [
            SeriesDef("area", np.nan, "float"),
            SeriesDef("spatial_unit_id", 0, "int32"),
            SeriesDef("land_class_id", 0, "int32"),
            SeriesDef("age", 0, "int32"),
            SeriesDef("species", 0, "int32"),
            SeriesDef("time_since_last_disturbance", 0, "int32"),
            SeriesDef("time_since_land_use_change", 0, "int32"),
            SeriesDef("last_disturbance_type", 0, "int32"),
            SeriesDef("enabled", 0, "int32"),
        ],
        nrows=n_rows,
        back_end=backend_type,
    )


def init_spinup_state(n_rows: int, backend_type: BackendType):
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

    return CBMVariables(
        {
            "pools": init_pools(n_rows, pool_names, backend_type),
            "flux": init_flux(n_rows, flux_names, backend_type),
            "parameters": init_parameters(n_rows, backend_type),
            "state": init_state(n_rows, backend_type),
        }
    )
