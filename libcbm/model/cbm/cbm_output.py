from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import series
from libcbm.storage.backends import BackendType


def _get_disturbance_type_map_func(disturbance_type_map):
    def disturbance_type_map_func(dist_id):
        if dist_id <= 0:
            return dist_id
        else:
            return disturbance_type_map[dist_id]

    return disturbance_type_map_func


def _add_timestep_series(timestep: int, dataframe: DataFrame) -> DataFrame:
    dataframe.add_column(
        series.range(
            "identifier", 0, dataframe.n_rows, 1, "int", dataframe.backend_type
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
    def __init__(
        self,
        density: bool = False,
        classifier_map: dict[int, str] = None,
        disturbance_type_map: dict[int, str] = None,
        backend_type: BackendType = BackendType.numpy,
        backend_params: dict = None,
    ):
        """Create storage and a function for complete simulation results.  The
        function return value can be passed to :py:func:`simulate` to track
        simulation results.

        Args:
            density (bool, optional): if set to true pool and flux indicators
                will be computed as area densities (tonnes C/ha). By default,
                pool and flux outputs are computed as mass (tonnes C) based on
                the area of each stand. Defaults to False.
            classifier_map (dict, optional): a classifier map for subsituting
                the internal classifier id values with classifier value names.
                If specified, the names associated with each id in the map are
                the values in the  the classifiers result DataFrame  If set to
                None the id values will be returned.
            disturbance_type_map (dict, optional): a disturbance type map for
                subsituting the internally defined disturbance type id with
                names or other ids in the parameters and state tables.  If set
                to none no substitution will occur.
            backend_type (BackendType, optional): the storage backend for the
                output, one of the values of
                :py:class:`libcbm.storage.backends.BackendType`. Defaults to
                `BackendType.numpy` meaning simulation results will be stored
                in memory.
            backend_params (dict): may be required depending on the specified
                backend_type, but not required by default. Defaults to None
        """
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
    def pools(self) -> DataFrame:
        return self._pools

    @property
    def flux(self) -> DataFrame:
        return self._flux

    @property
    def state(self) -> DataFrame:
        return self._state

    @property
    def classifiers(self) -> DataFrame:
        return self._classifiers

    @property
    def parameters(self) -> DataFrame:
        return self._parameters

    @property
    def area(self) -> DataFrame:
        return self._area

    def append_simulation_result(self, timestep: int, cbm_vars: CBMVariables):
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

        timestep_state = cbm_vars.state.copy()
        timestep_params = cbm_vars.parameters.copy()
        if self._disturbance_type_map:

            dist_map_func = _get_disturbance_type_map_func(
                self._disturbance_type_map
            )
            timestep_state["last_disturbance_type"].assign(
                timestep_state["last_disturbance_type"].map(dist_map_func),
                allow_type_change=True,
            )

            timestep_params["disturbance_type"].assign(
                timestep_params["disturbance_type"].map(dist_map_func),
                allow_type_change=True,
            )

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
                self._classifier_map
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
