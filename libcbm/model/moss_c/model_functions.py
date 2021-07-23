import numpy as np
from enum import IntEnum
from types import SimpleNamespace
from libcbm.model.moss_c.pools import Pool

import numba
import numba.typed


class SpinupState(IntEnum):
    AnnualProcesses = 1,
    HistoricalEvent = 2,
    LastPassEvent = 3,
    End = 4


def expand_matrix(mat, initialize_identity=True):
    n_mats = len(mat[0][2])
    n_rows = len(mat)
    out_rows = [float(mat[r][0]) for r in range(0, n_rows)]
    out_cols = [float(mat[r][1]) for r in range(0, n_rows)]
    out_values = [
        np.array([mat[r][2]])
        if np.isscalar(mat[r][2])
        else np.array(mat[r][2])
        for r in range(0, n_rows)]
    if initialize_identity:
        identity_set = {int(p) for p in Pool}

        for r in range(0, n_rows):
            if out_rows[r] == out_cols[r]:
                identity_set.remove(int(out_rows[r]))
    else:
        identity_set = []
    return __expand_matrix(
        n_mats, n_rows,
        numba.typed.List(out_rows),
        numba.typed.List(out_cols),
        numba.typed.List(out_values),
        numba.typed.List(identity_set))


@numba.njit
def __expand_matrix(n_mats, n_rows, out_rows, out_cols, out_values,
                    identity_set):

    n_output_rows = n_rows + len(identity_set)
    output = [np.zeros(shape=(n_output_rows, 3)) for _ in range(0, n_mats)]
    for i in range(0, n_mats):
        for r, pool in enumerate(identity_set):
            output[i][r][0] = pool
            output[i][r][1] = pool
            output[i][r][2] = 1.0
    for i in range(0, n_mats):
        for r in range(0, n_rows):
            r_offset = r + len(identity_set)
            output[i][r_offset][0] = out_rows[r]
            output[i][r_offset][1] = out_cols[r]
            if out_values[r].size == 1:
                output[i][r_offset][2] = out_values[r][0]
            else:
                output[i][r_offset][2] = out_values[r][i]

    return output


def compute(dll, pools, ops, op_indices, op_processes=None,
            flux=None, enabled=None):

    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        dll.set_op(op_id, [x for x in op],
                   np.ascontiguousarray(op_indices[:, i]))

    if flux is not None:
        dll.compute_flux(op_ids, op_processes, pools, flux, enabled)
    else:
        dll.compute_pools(op_ids, pools, enabled)
    for op_id in op_ids:
        dll.free_op(op_id)


def build_merch_vol_lookup(merch_volume):
    merch_vol_lookup = {int(i): {} for i in merch_volume.index}
    for _, row in merch_volume.iterrows():
        merch_vol_lookup[int(row.name)][int(row.age)] = float(row.volume)
    return merch_vol_lookup


def get_merch_vol(merch_vol_lookup, age, merch_vol_id):
    output = np.zeros(shape=age.shape, dtype=float)
    for i, age in np.ndenumerate(age):
        lookup = merch_vol_lookup[merch_vol_id[i]]
        if age in lookup:
            output[i] = lookup[age]
        else:
            output[i] = max(lookup, key=int)
    return output


def np_map(a, m, dtype):
    """Return the mapped value of a according to the dictionary m.
    Any values in a not present as a key in m will be unchanged.

    Args:
        a (numpy.ndarray): a numpy array
        m (dict): a dictionary to map values in the resulting array
        dtype (object): numpy type assigned as type of returned array
    Returns:
        numpy.ndarray: the numpy array with replaced mapped values
    """
    d = numba.typed.Dict()
    for k, v in m.items():
        d[k] = v
    out = np.ndarray(shape=a.shape, dtype=dtype)
    return _np_map(a, d, out)


@numba.njit
def _np_map(a, m, out):
    for index, value in np.ndenumerate(a):
        if value in m:
            out[index] = m[value]
    return out


def initialize_dm(disturbance_matrix_data):
    dm_data = disturbance_matrix_data
    proportions_valid = np.allclose(
        1.0,
        dm_data[["disturbance_type_id", "source", "proportion"]].groupby(
            ["disturbance_type_id", "source"]).sum())
    if not proportions_valid:
        raise ValueError("proportions in disturbance matrices do not sum to 1")

    identity_matrix = np.column_stack([
        np.array([int(p) for p in Pool], dtype=float),
        np.array([int(p) for p in Pool], dtype=float),
        np.repeat(1.0, len(Pool))])
    dm_list = [identity_matrix]
    dm_dist_type_index = {0: 0}
    for dist_type_id in dm_data.disturbance_type_id.unique():
        dm_values = dm_data[
            dm_data.disturbance_type_id == dist_type_id].copy()

        identity_set = {p for p in Pool}

        for _, row in dm_values.iterrows():
            if Pool[row.source] == Pool[row.sink]:
                identity_set.remove(Pool[row.source])

        dm_values = dm_values.append([
            {"disturbance_type_id": dist_type_id,
             "source": p.name,
             "sink": p.name,
             "proportion": 1.0} for p in identity_set],
            ignore_index=True)
        mat = np.column_stack([
            np.array([Pool[p] for p in dm_values.source], dtype=float),
            np.array([Pool[p] for p in dm_values.sink], dtype=float),
            dm_values.proportion.to_numpy()
        ])
        dm_dist_type_index[dist_type_id] = len(dm_list)
        dm_list.append(mat)

    return SimpleNamespace(
        dm_dist_type_index=dm_dist_type_index,
        dm_list=dm_list)


def to_numpy_namespace(df):
    return SimpleNamespace(**{
        col: df[col].to_numpy() for col in df.columns
    })


def _small_slow_diff(last_rotation_slow, this_rotation_slow):
    return abs((last_rotation_slow - this_rotation_slow)
               / (last_rotation_slow+this_rotation_slow)/2.0) < 0.001


@numba.jit
def advance_spinup_state(spinup_state, age, final_age, return_interval,
                         rotation_num, max_rotations, last_rotation_slow,
                         this_rotation_slow):

    out_state = spinup_state.copy()
    for i, state in np.ndenumerate(spinup_state):
        if state == SpinupState.AnnualProcesses:
            if age[i] >= return_interval[i]:
                small_slow_diff = _small_slow_diff(
                    last_rotation_slow[i], this_rotation_slow)
                if small_slow_diff | rotation_num[i] > max_rotations[i]:
                    out_state[i] = SpinupState.LastPassEvent
                else:
                    out_state[i] = SpinupState.HistoricalEvent
            else:
                out_state[i] = SpinupState.AnnualProcesses
        elif state == SpinupState.HistoricalEvent:
            out_state[i] = SpinupState.AnnualProcesses
        elif state == SpinupState.LastPassEvent:
            if age < final_age:
                out_state[i] = SpinupState.AnnualProcesses
            elif age >= final_age:
                out_state[i] = SpinupState.End
    return out_state
