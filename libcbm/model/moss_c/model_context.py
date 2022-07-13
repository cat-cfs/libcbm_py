import json


from libcbm.model.moss_c.pools import Pool
from libcbm.model.moss_c.pools import FLUX_INDICATORS
from libcbm.model.moss_c.merch_vol_lookup import MerchVolumeLookup
from libcbm.model.moss_c import model_functions
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
from libcbm.storage import series
from libcbm.storage.backends import BackendType
from libcbm.model.moss_c.model_functions import DMData


class ModelContext:
    def __init__(
        self,
        inventory: DataFrame,
        parameters: DataFrame,
        merch_volume: DataFrame,
        disturbance_matrices: DataFrame,
        backend_type: BackendType,
    ):
        self._backend_type = backend_type
        self._parameters = parameters
        self._n_stands = inventory.n_rows
        self._inventory = inventory
        self._dm_info = disturbance_matrices
        self._merch_vol_lookup = MerchVolumeLookup(merch_volume.to_pandas())
        self._dll = self._initialize_libcbm()
        self._pools = self._initialize_pools()
        self._flux = self._initialize_flux()
        self.state = self._initialize_model_state()
        self._disturbance_matrices = self._initialize_disturbance_data()

    @property
    def backend_type(self) -> BackendType:
        return self._backend_type

    @property
    def parameters(self) -> DataFrame:
        return self._parameters

    @property
    def n_stands(self) -> int:
        return self._n_stands

    @property
    def inventory(self) -> DataFrame:
        return self._inventory

    @property
    def merch_vol_lookup(self) -> MerchVolumeLookup:
        return self._merch_vol_lookup

    @property
    def dll(self) -> LibCBMWrapper:
        return self._dll

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
    def disturbance_matrices(self) -> DMData:
        return self._disturbance_matrices

    def _initialize_libcbm(self) -> LibCBMWrapper:
        libcbm_config = {
            "pools": [
                {"name": p.name, "id": int(p), "index": p_idx}
                for p_idx, p in enumerate(Pool)
            ],
            "flux_indicators": [
                {
                    "id": f_idx + 1,
                    "index": f_idx,
                    "process_id": f["process_id"],
                    "source_pools": [int(x) for x in f["source_pools"]],
                    "sink_pools": [int(x) for x in f["sink_pools"]],
                }
                for f_idx, f in enumerate(FLUX_INDICATORS)
            ],
        }
        return LibCBMWrapper(
            LibCBMHandle(
                resources.get_libcbm_bin_path(), json.dumps(libcbm_config)
            )
        )

    def _initialize_pools(self) -> DataFrame:
        pools = dataframe.numeric_dataframe(
            Pool.__members__.keys(), self.n_stands, self._backend_type
        )
        pools[Pool.Input.name].assign_all(1.0)
        return pools

    def _initialize_flux(self) -> DataFrame:
        return dataframe.numeric_dataframe(
            [x["name"] for x in FLUX_INDICATORS],
            self.n_stands,
            self._backend_type,
        )

    def _initialize_model_state(self) -> DataFrame:
        initial_age = series.allocate(
            "age", self.n_stands, 0, "int", back_end=self._backend_type
        )
        return dataframe.from_series_dict(
            dict(
                age=initial_age,
                merch_vol=self.merch_vol_lookup.get_merch_vol(
                    initial_age, self.parameters["merch_volume_id"]
                ),
                enabled=series.allocate(
                    "enabled",
                    self.n_stands,
                    1,
                    dtype="int32",
                    back_end=self._backend_type,
                ),
                disturbance_type=series.allocate(
                    "disturbance_type", self.n_stands, 0, "uintp"
                ),
            ),
            nrows=self.n_stands,
            back_end=self.backend_type,
        )

    def _initialize_disturbance_data(self) -> DMData:
        disturbance_matrices = model_functions.initialize_dm(
            self._dm_info.to_pandas()
        )
        historical_dist_types = self._inventory[
            "historical_disturbance_type_id"
        ].to_numpy()
        last_pass_dist_types = self._inventory[
            "last_pass_disturbance_type_id"
        ].to_numpy()
        self._inventory.add_column(
            series.from_numpy(
                "historical_dm_index",
                model_functions.np_map(
                    historical_dist_types,
                    disturbance_matrices.dm_dist_type_index,
                    dtype="int64",
                ),
            )
        )
        self._inventory.add_column(
            series.from_numpy(
                "last_pass_dm_index",
                model_functions.np_map(
                    last_pass_dist_types,
                    disturbance_matrices.dm_dist_type_index,
                    dtype="int64",
                ),
            )
        )
        return disturbance_matrices
