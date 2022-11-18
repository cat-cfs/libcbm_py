from enum import IntEnum
import numpy as np
from libcbm.storage.series import Series

import numba


class SpinupState(IntEnum):
    AnnualProcesses = 1
    HistoricalEvent = 2
    LastPassEvent = 3
    GrowToFinalAge = 4
    End = 5


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
            if age[i] >= (return_interval[i]):

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
