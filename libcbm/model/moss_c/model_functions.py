import pandas as pd
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
        dm_data[["disturbance_type_id", "source", "proportion"]]
        .groupby(["disturbance_type_id", "source"])
        .sum(),
    )
    if not proportions_valid:
        raise ValueError("proportions in disturbance matrices do not sum to 1")

    identity_matrix = np.column_stack(
        [
            np.array([int(p) for p in Pool], dtype=float),
            np.array([int(p) for p in Pool], dtype=float),
            np.repeat(1.0, len(Pool)),
        ]
    )
    dm_list = [identity_matrix]
    dm_dist_type_index = {0: 0}
    for dist_type_id in dm_data.disturbance_type_id.unique():
        dm_values = dm_data[dm_data.disturbance_type_id == dist_type_id].copy()

        identity_set = {p for p in Pool}

        for _, row in dm_values.iterrows():
            if Pool[row.source] == Pool[row.sink]:
                identity_set.remove(Pool[row.source])

        dm_values = pd.concat(
            [
                dm_values,
                pd.DataFrame(
                    data=[
                        {
                            "disturbance_type_id": dist_type_id,
                            "source": p.name,
                            "sink": p.name,
                            "proportion": 1.0,
                        }
                        for p in identity_set
                    ]
                ),
            ]
        )
        mat = np.column_stack(
            [
                np.array([Pool[p] for p in dm_values.source], dtype=float),
                np.array([Pool[p] for p in dm_values.sink], dtype=float),
                dm_values.proportion.to_numpy(),
            ]
        )
        dm_dist_type_index[dist_type_id] = len(dm_list)
        dm_list.append(mat)

    return SimpleNamespace(
        dm_dist_type_index=dm_dist_type_index, dm_list=dm_list
    )


def to_numpy_namespace(df):
    return SimpleNamespace(**{col: df[col].to_numpy() for col in df.columns})


@numba.njit()
def _small_slow_diff(last_rotation_slow, this_rotation_slow):
    return (
        abs(
            (last_rotation_slow - this_rotation_slow)
            / (last_rotation_slow + this_rotation_slow)
            / 2.0
        )
        < 0.001
    )


def advance_spinup_state(
    spinup_state,
    age,
    final_age,
    return_interval,
    rotation_num,
    max_rotations,
    last_rotation_slow,
    this_rotation_slow,
):

    out_state = spinup_state.copy()
    _advance_spinup_state(
        age.shape[0],
        spinup_state,
        age,
        final_age,
        return_interval,
        rotation_num,
        max_rotations,
        last_rotation_slow,
        this_rotation_slow,
        out_state,
    )
    return out_state


@numba.njit()
def _advance_spinup_state(
    n_stands,
    spinup_state,
    age,
    final_age,
    return_interval,
    rotation_num,
    max_rotations,
    last_rotation_slow,
    this_rotation_slow,
    out_state,
):

    for i in range(0, n_stands):
        state = spinup_state[i]
        if state == SpinupState.AnnualProcesses:
            if age[i] >= (return_interval[i] - 1):

                small_slow_diff = (
                    _small_slow_diff(
                        last_rotation_slow[i], this_rotation_slow[i]
                    )
                    if (last_rotation_slow[i] > 0)
                    | (this_rotation_slow[i] > 0)
                    else False
                )
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
