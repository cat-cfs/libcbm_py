import numpy as np
import pandas as pd
import json
from contextlib import contextmanager
from libcbm.wrapper import libcbm_operation
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources


@contextmanager
def create_model(pools, flux_indicators):

    libcbm_config = {
        "pools": [
            {'name': p, 'id': p_idx, 'index': p_idx}
            for p, p_idx in pools.items()],
        "flux_indicators": [
            {
                "id": f_idx + 1,
                "index": f_idx,
                "process_id": f["process_id"],
                "source_pools": [int(x) for x in f["source_pools"]],
                "sink_pools": [int(x) for x in f["sink_pools"]],
            } for f_idx, f in enumerate(flux_indicators)]
        }

    with LibCBMHandle(
            resources.get_libcbm_bin_path(),
            json.dumps(libcbm_config)
    ) as handle:
        yield ModelHandle(
            LibCBMWrapper(handle), pools, flux_indicators
        )


class ModelHandle():

    def __init__(self, wrapper, pools, flux_indicators):
        self.wrapper = wrapper
        self.pools = pools
        self.flux_indicators = flux_indicators

    def allocate_pools(self, n):
        return np.zeros(shape=(n, len(self.pools)))

    def allocate_flux(self, n):
        return np.zeros(shape=(n, len(self.flux_indicators)))

    def _matrix_rc(self, value):
        return libcbm_operation.Operation(
            self.wrapper,
            libcbm_operation.OperationFormat.RepeatingCoordinates,
            value)

    def _matrix_list(self, value):
        return libcbm_operation.Operation(
            self.wrapper,
            libcbm_operation.OperationFormat.MatrixList,
            value)

    def create_operation(self, matrices, fmt):
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

    def compute(self, pools, flux, operations, op_processes, enabled):
        libcbm_operation.compute(
            dll=self.wrapper,
            pools=pools,
            operations=operations,
            op_processes=op_processes,
            flux=flux,
            enabled=enabled)

    def create_output_processor(self, type="in_memory"):
        return ModelOutputProcessor(self)


class ModelOutputProcessor():

    def __init__(self, model_handle: ModelHandle):
        self.model_handle = model_handle
        self.pools = pd.DataFrame()
        self.flux = pd.DataFrame()

    def append_results(self, t, pools, flux):
        pools_t = pd.DataFrame(
            columns=self.model_handle.pools.keys(),
            data=pools.copy())
        pools_t.insert(0, "timestep", t)
        self.pools = self.pools.append(pools_t)

        flux_t = pd.DataFrame(
            columns=[
                x["name"] for x in
                self.model_handle.flux_indicators],
            data=flux.copy()
        )
        flux_t.insert(0, "timestep", t)
        self.flux = self.flux.append(flux_t)
