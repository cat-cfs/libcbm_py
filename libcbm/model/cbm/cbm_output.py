from __future__ import annotations
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import series
from libcbm.storage.backends import BackendType


def _add_timestep_series(timestep: int, dataframe: DataFrame) -> DataFrame:
    dataframe.add_column(
        series.range(
            "identifier",
            1,
            dataframe.n_rows + 1,
            1,
            "int64",
            dataframe.backend_type,
        ),
        0,
    )
    dataframe.add_column(
        series.allocate(
            "timestep",
            dataframe.n_rows,
            timestep,
            "int",
            dataframe.backend_type,
        ),
        1,
    )
    return dataframe


def _concat_timestep_results(
    timestep: int,
    running_result: DataFrame,
    timestep_result: DataFrame,
    backend_type: BackendType,
) -> DataFrame:
    _add_timestep_series(timestep, timestep_result)

    return dataframe.concat_data_frame(
        [running_result, timestep_result], backend_type
    )


class CBMOutput:
    """
    Initialize CBMOutput

    Args:
        density (bool, optional): if set to true pool and flux indicators
            will be computed as area densities (tonnes C/ha). By default,
            pool and flux outputs are computed as mass (tonnes C) based on
            the area of each stand. Defaults to False.
        classifier_map (dict[int, str], optional): a classifier map for
            subsituting the internal classifier id values with classifier
            value names. If specified, the names associated with each id
            in the map are the values in the the classifiers result
            DataFrame. If set to None the id values will be returned.
        disturbance_type_map (dict[int, str], optional): a disturbance
            type map for subsituting the internally defined disturbance
            type id with names or other ids in the parameters and state
            tables.  If set to none no substitution will occur.
        backend_type (BackendType, optional): the storage backend for the
            output, one of the values of
            :py:class:`libcbm.storage.backends.BackendType`. Defaults to
            `BackendType.numpy` meaning simulation results will be stored
            in memory.
    """

    def __init__(
        self,
        density: bool = False,
        classifier_map: dict[int, str] = None,
        disturbance_type_map: dict[int, str] = None,
        backend_type: BackendType = BackendType.numpy,
    ):
        self._density = density
        self._disturbance_type_map = disturbance_type_map
        self._classifier_map = classifier_map
        self._backend_type = backend_type
        self._pools: DataFrame = None
        self._flux: DataFrame = None
        self._state: DataFrame = None
        self._classifiers: DataFrame = None
        self._parameters: DataFrame = None
        self._area: DataFrame = None

    @property
    def density(self) -> bool:
        return self._density

    @property
    def disturbance_type_map(self) -> dict[int, str]:
        """get this instance's disturbance type map"""
        return self._disturbance_type_map

    @property
    def classifier_map(self) -> dict[int, str]:
        """get this instance's clasifier map"""
        return self._classifier_map

    @property
    def backend_type(self) -> BackendType:
        """get this instance's backend type"""
        return self._backend_type

    @property
    def pools(self) -> DataFrame:
        """get all accumulated pool results"""
        return self._pools

    @property
    def flux(self) -> DataFrame:
        """get all accumulated flux results"""
        return self._flux

    @property
    def state(self) -> DataFrame:
        """get all accumulated state results"""
        return self._state

    @property
    def classifiers(self) -> DataFrame:
        """get all accumulated clasifier results"""
        return self._classifiers

    @property
    def parameters(self) -> DataFrame:
        """get all accumulated parameter results"""
        return self._parameters

    @property
    def area(self) -> DataFrame:
        """get all accumulated area results"""
        return self._area

    def append_simulation_result(self, timestep: int, cbm_vars: CBMVariables):
        """Append simulation resuls

        Args:
            timestep (int): the timestep corresponding to the results
            cbm_vars (CBMVariables): The cbm vars for the timestep
        """
        timestep_pools = (
            cbm_vars.pools.copy()
            if self._density
            else cbm_vars.pools.multiply(cbm_vars.inventory["area"])
        )
        self._pools = _concat_timestep_results(
            timestep, self._pools, timestep_pools, self._backend_type
        )

        if cbm_vars.flux is not None and cbm_vars.flux.n_rows > 0:
            timestep_flux = (
                cbm_vars.flux.copy()
                if self._density
                else cbm_vars.flux.multiply(cbm_vars.inventory["area"])
            )
            self._flux = _concat_timestep_results(
                timestep, self._flux, timestep_flux, self._backend_type
            )

        if self._disturbance_type_map:
            timestep_state_data = {
                c: cbm_vars.state[c].copy() for c in cbm_vars.state.columns
            }
            timestep_state_data["last_disturbance_type"] = timestep_state_data[
                "last_disturbance_type"
            ].map(self._disturbance_type_map)
            timestep_state = dataframe.from_series_dict(
                timestep_state_data,
                cbm_vars.state.n_rows,
                cbm_vars.state.backend_type,
            )

            timestep_params_data = {
                c: cbm_vars.parameters[c].copy()
                for c in cbm_vars.parameters.columns
            }

            timestep_params_data["disturbance_type"] = timestep_params_data[
                "disturbance_type"
            ].map(self._disturbance_type_map)
            timestep_params = dataframe.from_series_dict(
                timestep_params_data,
                cbm_vars.parameters.n_rows,
                cbm_vars.parameters.backend_type,
            )
        else:
            timestep_state = cbm_vars.state.copy()
            timestep_params = cbm_vars.parameters.copy()

        self._state = _concat_timestep_results(
            timestep, self._state, timestep_state, self._backend_type
        )

        self._parameters = _concat_timestep_results(
            timestep, self._parameters, timestep_params, self._backend_type
        )

        if self._classifier_map is None:
            self._classifiers = _concat_timestep_results(
                timestep,
                self._classifiers,
                cbm_vars.classifiers.copy(),
                self._backend_type,
            )
        else:
            timestep_classifiers = cbm_vars.classifiers.copy()
            timestep_classifiers = timestep_classifiers.map(
                self.classifier_map
            )
            self._classifiers = _concat_timestep_results(
                timestep,
                self._classifiers,
                timestep_classifiers,
                self._backend_type,
            )
        self._area = _concat_timestep_results(
            timestep,
            self._area,
            dataframe.from_series_list(
                [cbm_vars.inventory["area"]],
                nrows=cbm_vars.inventory.n_rows,
                back_end=self._backend_type,
            ),
            self._backend_type,
        )
