import numpy as np
import pandas as pd
import numba
from numba.core import types
from numba.typed import Dict
from libcbm.model.model_definition.model import CBMModel


def disturbance(
    model: CBMModel,
    disturbance_matrices: pd.DataFrame,
    dm_associations: pd.DataFrame,
) -> tuple[list, Dict]:
    matrix_data_by_dmid: dict[int, list[list]] = {}
    dmid_index = Dict.empty(key_type=types.int64, value_type=types.int64)

    dmid = disturbance_matrices["disturbance_matrix_id"].to_numpy()
    source = disturbance_matrices["source_pool"].to_list()
    sink = disturbance_matrices["sink_pool"].to_list()
    proportion = disturbance_matrices["proportion"].to_numpy()
    for i in range(dmid.shape[0]):
        dmid_i = dmid[i]
        if dmid_i not in matrix_data_by_dmid:
            matrix_data_by_dmid[dmid_i] = []
        matrix_data_by_dmid[dmid_i].append([source[i], sink[i], proportion[i]])
    dmids = list(matrix_data_by_dmid.keys())
    for i_dmid, dmid in enumerate(dmids):
        pool_set = set(model.pool_names)
        dmid_index[dmid] = i_dmid + 1
        for row in matrix_data_by_dmid[dmid]:
            pool_set.discard(row[0])
        for pool in pool_set:
            matrix_data_by_dmid[dmid].append([pool, pool, 1.0])

    _dm_op_index = Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.int64,
    )
    _dm_op_index = _build_dm_op_index(
        _dm_op_index,
        dmid_index,
        dm_associations["disturbance_matrix_id"].to_numpy(dtype="int"),
        dm_associations["disturbance_type_id"].to_numpy(dtype="int"),
        dm_associations["spatial_unit_id"].to_numpy(dtype="int"),
        dm_associations["sw_hw"].to_numpy(dtype="int"),
    )
    # append the null matrix
    matrix_list = [[[p, p, 1.0] for p in model.pool_names]] + list(
        matrix_data_by_dmid.values()
    )
    return (matrix_list, _dm_op_index)


@numba.njit()
def _build_dm_op_index(
    _dm_op_index: Dict,
    dmid_idx: Dict,
    disturbance_matrix_id: np.ndarray,
    disturbance_type_id: np.ndarray,
    spatial_unit_ids: np.ndarray,
    sw_hw: np.ndarray,
):
    for i in range(disturbance_type_id.shape[0]):
        dist_type = disturbance_type_id[i]
        spuid = spatial_unit_ids[i]
        _sw_hw = sw_hw[i]
        key = (dist_type, spuid, _sw_hw)
        dm_idx = dmid_idx[disturbance_matrix_id[i]]
        _dm_op_index[key] = dm_idx

    return _dm_op_index
