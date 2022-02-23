import json
import numpy as np
from typing import ContextManager
from contextlib import contextmanager
from libcbm.wrapper import libcbm_operation
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources
from libcbm.storage.dataframe import Series
from libcbm.storage import dataframe
from libcbm.storage.backends import BackendType
from libcbm.storage import dataframe_functions


class ModelVars:
    def __init__(
        self,
        size: int,
        pool_names: list[str],
        flux_names: list[str],
        backend_type: BackendType = None,
    ):
        self.pools = dataframe.numeric_dataframe(
            pool_names, size, 0, backend_type
        )
        self.flux = dataframe.numeric_dataframe(
            flux_names, size, 0, backend_type
        )


class ModelHandle:
    def __init__(
        self,
        wrapper: LibCBMWrapper,
        pools: dict[str, int],
        flux_indicators: list[dict],
    ):
        self.wrapper = wrapper
        self.pools = pools
        self.flux_indicators = flux_indicators

    def allocate_model_vars(self, n: int):
        return ModelVars(
            n,
            list(self.pools.keys()),
            [x["name"] for x in self.flux_indicators],
        )

    def _matrix_rc(self, value: list) -> libcbm_operation.Operation:
        return libcbm_operation.Operation(
            self.wrapper,
            libcbm_operation.OperationFormat.RepeatingCoordinates,
            value,
        )

    def _matrix_list(self, value: list) -> libcbm_operation.Operation:
        return libcbm_operation.Operation(
            self.wrapper, libcbm_operation.OperationFormat.MatrixList, value
        )

    def create_operation(
        self, matrices: list, fmt: str
    ) -> libcbm_operation.Operation:
        if fmt == "repeating_coordinates":
            pool_id_mat = [
                [self.pools[row[0]], self.pools[row[1]], row[2]]
                for row in matrices
            ]
            return self._matrix_rc(pool_id_mat)
        elif fmt == "matrix_list":
            mat_list = []
            for mat in matrices:
                mat_len = len(mat)
                np_mat = np.zeros(shape=(mat_len, 3))
                for i_entry, entry in enumerate(mat):
                    np_mat[i_entry, 0] = self.pools[entry[0]]
                    np_mat[i_entry, 1] = self.pools[entry[1]]
                    np_mat[i_entry, 2] = entry[2]
                mat_list.append(np_mat)
            return self._matrix_list(mat_list)
        else:
            raise ValueError("unknown format")

    def compute(
        self,
        model_vars: ModelVars,
        operations: list[libcbm_operation.Operation],
        op_processes: list[int],
        enabled: Series,
    ) -> None:

        libcbm_operation.compute(
            dll=self.wrapper,
            pools=model_vars.pools,
            operations=operations,
            op_processes=[int(o) for o in op_processes],
            flux=model_vars.flux,
            enabled=enabled.astype(int) if enabled is not None else None,
        )

    def create_output_processor(
        self, type="in_memory"
    ) -> "ModelOutputProcessor":
        return ModelOutputProcessor(self)


class ModelOutputProcessor:
    def __init__(self, model_handle: ModelHandle):
        self.model_handle = model_handle
        self.pools = None
        self.flux = None

    def append_results(self, t: int, model_vars: ModelVars):
        pools_t = model_vars.pools.copy()
        pools_t.add_column(Series("timestep", t, "int"), 0)
        dataframe_functions.concat_data_frame([self.pools, pools_t])

        flux_t = model_vars.flux.copy()
        flux_t.add_column(Series("timestep", t, "int"), 0)
        dataframe_functions.concat_data_frame([self.flux, flux_t])


@contextmanager
def create_model(
    pools: dict[str, int], flux_indicators: list[dict]
) -> ContextManager[ModelHandle]:

    libcbm_config = {
        "pools": [
            {"name": p, "id": p_idx, "index": p_idx}
            for p, p_idx in pools.items()
        ],
        "flux_indicators": [
            {
                "id": f_idx + 1,
                "index": f_idx,
                "process_id": f["process_id"],
                "source_pools": [int(x) for x in f["source_pools"]],
                "sink_pools": [int(x) for x in f["sink_pools"]],
            }
            for f_idx, f in enumerate(flux_indicators)
        ],
    }

    with LibCBMHandle(
        resources.get_libcbm_bin_path(), json.dumps(libcbm_config)
    ) as handle:
        yield ModelHandle(LibCBMWrapper(handle), pools, flux_indicators)
