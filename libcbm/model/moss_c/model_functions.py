import numpy as np
from enum import IntEnum
from types import SimpleNamespace
from libcbm.model.moss_c.pools import Pool

import numba
import numba.typed
import numba.types


class SpinupState(IntEnum):
    AnnualProcesses = 1
    HistoricalEvent = 2
    LastPassEvent = 3
    GrowToFinalAge = 4
    End = 5


def expand_matrix(mat, identity_set=None):
    """Expand a coordinate-matrix-like object into a list of coordinate matrix
    triples.  The array length of the 3rd element of all rows should be equal.

    Example intput format::

        [
            [r0, c0, [v0_0,v0_1,...,v0_n]],
            [r1, c1, [v1_0,v1_1,...,v1_n]],
            ...
            [rk, ck, [vk_0,vk_1,...,vk_n]],
        ]

    Example output format::

        [
          [
            [r0, c0, v0_0],
            [r0, c0, v0_1],
            ...
            [r0, c0, v0_n]
          ],
          [
            [r1, c1, v1_0],
            [r1, c1, v1_1],
            ...
            [r1, c1, v1_n]
          ],
            [rk, ck, vk_0],
            [rk, ck, vk_1],
            ...
            [rk, ck, vk_n]
          ]
        ]

    Args:
        mat (list): a list of coordinate-matrix-like triples
        identity_set (iterable, optional): If specified, all values where the
            row coordinate is equal to the column coordinate that are not
            specified in the mat parameter are assigned a value of 1.
            Defaults to None.

    Returns:
        list: list of numpy.ndarray coordinate matrix triples
    """
    n_mats = len(mat[0][2])
    n_rows = len(mat)
    out_rows = [float(mat[r][0]) for r in range(0, n_rows)]
    out_cols = [float(mat[r][1]) for r in range(0, n_rows)]
    out_values = [
        np.array([mat[r][2]])
        if np.isscalar(mat[r][2])
        else np.array(mat[r][2])
        for r in range(0, n_rows)]

    n_output_rows = n_rows

    if identity_set is not None:
        identity_set = set(identity_set)
        for r in range(0, n_rows):
            if out_rows[r] == out_cols[r]:
                identity_set.remove(int(out_rows[r]))
        n_output_rows += len(identity_set)
        identity_set = np.array(list(identity_set), dtype=float)
    else:
        identity_set = np.array([], dtype=float)

    output = [np.zeros(shape=(n_output_rows, 3)) for _ in range(0, n_mats)]
    return __expand_matrix(
        n_mats, n_rows,
        numba.typed.List(out_rows),
        numba.typed.List(out_cols),
        numba.typed.List(out_values),
        identity_set,
        numba.typed.List(output))


@numba.njit()
def __expand_matrix(n_mats, n_rows, out_rows, out_cols, out_values,
                    identity_set, output):

    for r, pool in enumerate(identity_set):
        for i in range(0, n_mats):
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
    """Compute pool flows and optionally track the fluxes

    see the methods in :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        dll (LibCBMWrapper): instance of libcbm wrapper
        pools (pandas.DataFrame): pools dataframe (stands by pools)
        ops (list): list of list of coordinate matrices
        op_indices (np.ndarray): matrix of op indices
        op_processes (iterable, optional): flux indicator op processes.
            Required if flux arg is specified. Defaults to None.
        flux (pandas.DataFrame, optional): Flux indicators dataframe
            (stands by flux-indicator). If not specified, no fluxes are
            tracked. Defaults to None.
        enabled (numpy.ndarray, optional): Flag array of length n-stands
            indicating whether or not to include corresponding rows in
            computation. If set to None, all records are included.
            Defaults to None.
    """
    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        dll.set_op(op_id, op, np.ascontiguousarray(op_indices[:, i]))

    if flux is not None:
        dll.compute_flux(op_ids, op_processes, pools, flux, enabled)
    else:
        dll.compute_pools(op_ids, pools, enabled)
    for op_id in op_ids:
        dll.free_op(op_id)


def np_map(a, m, dtype):
    """Return the mapped value of a according to the dictionary m.
    The occurence of any value in a that is not present as a key
    in m will result in a value error.

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
    out = np.empty_like(a, dtype=dtype)
    return _np_map(a, d, out)


@numba.njit()
def _np_map(a, m, out):
    for index, value in np.ndenumerate(a):
        if value in m:
            out[index] = m[value]
        else:
            raise ValueError("value not present in supplied array")
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


@numba.njit()
def _small_slow_diff(last_rotation_slow, this_rotation_slow):
    return abs(
        (last_rotation_slow - this_rotation_slow)
        / (last_rotation_slow + this_rotation_slow) / 2.0) < 0.001


def advance_spinup_state(spinup_state, age, final_age, return_interval,
                         rotation_num, max_rotations, last_rotation_slow,
                         this_rotation_slow):

    out_state = spinup_state.copy()
    _advance_spinup_state(age.shape[0], spinup_state, age, final_age,
                          return_interval, rotation_num, max_rotations,
                          last_rotation_slow, this_rotation_slow, out_state)
    return out_state


@numba.njit()
def _advance_spinup_state(n_stands, spinup_state, age, final_age,
                          return_interval, rotation_num, max_rotations,
                          last_rotation_slow, this_rotation_slow, out_state):

    for i in range(0, n_stands):
        state = spinup_state[i]
        if state == SpinupState.AnnualProcesses:
            if age[i] >= (return_interval[i] - 1):

                small_slow_diff = (
                    _small_slow_diff(
                        last_rotation_slow[i], this_rotation_slow[i])
                    if (last_rotation_slow[i] > 0) |
                       (this_rotation_slow[i] > 0)
                    else False)
                if small_slow_diff | (rotation_num[i] >= max_rotations[i]):
                    out_state[i] = SpinupState.LastPassEvent
                else:
                    out_state[i] = SpinupState.HistoricalEvent
            else:
                out_state[i] = SpinupState.AnnualProcesses
        elif state == SpinupState.HistoricalEvent:
            out_state[i] = SpinupState.AnnualProcesses
        elif state == SpinupState.LastPassEvent:
            if age[i] < final_age[i]:
                out_state[i] = SpinupState.GrowToFinalAge
            elif age[i] >= final_age[i]:
                out_state[i] = SpinupState.End
        elif state == SpinupState.GrowToFinalAge:
            if age[i] < final_age[i]:
                out_state[i] = SpinupState.GrowToFinalAge
            else:
                out_state[i] = SpinupState.End
    return out_state
