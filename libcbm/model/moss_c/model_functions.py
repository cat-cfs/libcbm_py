import pandas as pd
import numpy as np
from enum import IntEnum
from libcbm.model.moss_c.pools import Pool

import numba
import numba.typed
import numba.types
from libcbm.storage.series import Series


class SpinupState(IntEnum):
    AnnualProcesses = 1
    HistoricalEvent = 2
    LastPassEvent = 3
    GrowToFinalAge = 4
    End = 5


def np_map(a: np.ndarray, m: dict, dtype: str):
    """Return the mapped value of a according to the dictionary m.
    The occurence of any value in a that is not present as a key
    in m will result in a value error.

    Args:
        a (numpy.ndarray): a numpy array
        m (dict): a dictionary to map values in the resulting array
        dtype (str): numpy type assigned as type of returned array
    Returns:
        numpy.ndarray: the numpy array with replaced mapped values
    """
    d = numba.typed.Dict()
    for k, v in m.items():
        d[k] = v
    out = np.empty_like(a, dtype=dtype)
    return _np_map(a, d, out)


@numba.njit()
def _np_map(a: np.ndarray, m: numba.typed.Dict, out: np.ndarray):
    for index, value in np.ndenumerate(a):
        if value in m:
            out[index] = m[value]
        else:
            raise ValueError("value not present in supplied array")
    return out


class DMData:
    def __init__(
        self, dm_dist_type_index: dict[int, int], dm_list: list[np.ndarray]
    ):
        self._dm_dist_type_index = dm_dist_type_index
        self._dm_list = dm_list

    @property
    def dm_dist_type_index(self) -> dict[int, int]:
        return self._dm_dist_type_index

    @property
    def dm_list(self) -> list[np.ndarray]:
        return self._dm_list


def initialize_dm(dm_data: pd.DataFrame) -> DMData:
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
    for dist_type_id in dm_data["disturbance_type_id"].unique():
        dm_values: pd.DataFrame = dm_data[
            dm_data["disturbance_type_id"] == dist_type_id
        ].copy()

        identity_set = {p for p in Pool}

        for _, row in dm_values.iterrows():
            if Pool[row["source"]] == Pool[row["sink"]]:
                identity_set.remove(Pool[row["source"]])

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
        mat: np.ndarray = np.column_stack(
            [
                np.array([Pool[p] for p in dm_values["source"]], dtype=float),
                np.array([Pool[p] for p in dm_values["sink"]], dtype=float),
                dm_values["proportion"].to_numpy(),
            ]
        )
        dm_dist_type_index[dist_type_id] = len(dm_list)
        dm_list.append(mat)

    return DMData(dm_dist_type_index=dm_dist_type_index, dm_list=dm_list)


@numba.njit()
def _small_slow_diff(
    last_rotation_slow: np.ndarray, this_rotation_slow: np.ndarray
) -> np.ndarray:
    return (
        abs(
            (last_rotation_slow - this_rotation_slow)
            / (last_rotation_slow + this_rotation_slow)
            / 2.0
        )
        < 0.001
    )


def advance_spinup_state(
    spinup_state: Series,
    age: Series,
    final_age: Series,
    return_interval: Series,
    rotation_num: Series,
    max_rotations: Series,
    last_rotation_slow: Series,
    this_rotation_slow: Series,
) -> np.ndarray:

    out_state = spinup_state.copy().to_numpy()
    _advance_spinup_state(
        age.length,
        spinup_state.to_numpy(),
        age.to_numpy(),
        final_age.to_numpy(),
        return_interval.to_numpy(),
        rotation_num.to_numpy(),
        max_rotations.to_numpy(),
        last_rotation_slow.to_numpy(),
        this_rotation_slow.to_numpy(),
        out_state,
    )
    return out_state


@numba.njit()
def _advance_spinup_state(
    n_stands: int,
    spinup_state: np.ndarray,
    age: np.ndarray,
    final_age: np.ndarray,
    return_interval: np.ndarray,
    rotation_num: np.ndarray,
    max_rotations: np.ndarray,
    last_rotation_slow: np.ndarray,
    this_rotation_slow: np.ndarray,
    out_state: np.ndarray,
) -> np.ndarray:

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
