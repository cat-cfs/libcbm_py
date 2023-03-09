from enum import IntEnum
import numpy as np
from libcbm.storage.series import Series

import numba


class SpinupState(IntEnum):
    """The possible spinup states for stands during spinup"""

    AnnualProcesses = 1
    HistoricalEvent = 2
    LastPassEvent = 3
    GrowToFinalAge = 4
    Delay = 5
    End = 6


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
    delay_step: Series,
    final_age: Series,
    delay: Series,
    return_interval: Series,
    rotation_num: Series,
    min_rotations: Series,
    max_rotations: Series,
    last_rotation_slow: Series,
    this_rotation_slow: Series,
    enabled: Series,
) -> np.ndarray:
    """Run the vectorized spinup finite state machine.  Passed values are not
    modified by this function.

    Args:
        spinup_state (Series): The current spinup state value. See
            :py:class:`SpinupState`
        age (Series): The age.
        delay_step (Series): The spinup delay step, when in the Delay state
        final_age (Series): The final age at the end of spinup, which is also
            known as the inventory age in CBM
        delay (Series): The number of delay steps to perform in spinup.
        return_interval (Series): The number of years between historical
            disturbance rotations.
        rotation_num (Series): The number of rotations already performed
        min_rotations (Series): The minimum number of rotations to perform
        max_rotations (Series): The maximum number of rotations to perform
        last_rotation_slow (Series): The sum of Slow C pools at the end of the
            last rotation, prior to historical disturbance.
        this_rotation_slow (Series): The current sum of Slow C pools.
        enabled (Series): A boolean flag indicating the corresponding stands
            are finished spinup (when 0), or spinup is ongoing (when 1)

    Returns:
        np.ndarray: The array of updated SpinupState.
    """
    out_state = spinup_state.copy().to_numpy()
    _advance_spinup_state(
        age.length,
        spinup_state.to_numpy(),
        age.to_numpy(),
        delay_step.to_numpy(),
        final_age.to_numpy(),
        delay.to_numpy(),
        return_interval.to_numpy(),
        rotation_num.to_numpy(),
        min_rotations.to_numpy(),
        max_rotations.to_numpy(),
        last_rotation_slow.to_numpy(),
        this_rotation_slow.to_numpy(),
        enabled.to_numpy(),
        out_state,
    )
    return out_state


@numba.njit()
def _advance_spinup_state(
    n_stands: int,
    spinup_state: np.ndarray,
    age: np.ndarray,
    delay_step: np.ndarray,
    final_age: np.ndarray,
    delay: np.ndarray,
    return_interval: np.ndarray,
    rotation_num: np.ndarray,
    min_rotations: np.ndarray,
    max_rotations: np.ndarray,
    last_rotation_slow: np.ndarray,
    this_rotation_slow: np.ndarray,
    enabled: np.ndarray,
    out_state: np.ndarray,
) -> np.ndarray:
    for i in range(0, n_stands):
        state = spinup_state[i]
        if not enabled[i]:
            out_state[i] = SpinupState.End
            continue
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
                if ((rotation_num[i] > min_rotations[i]) & small_slow_diff) | (
                    rotation_num[i] >= max_rotations[i]
                ):
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
                if delay[i] > 0:
                    out_state[i] = SpinupState.Delay
                else:
                    out_state[i] = SpinupState.End
        elif state == SpinupState.Delay:
            if delay_step[i] < delay[i]:
                out_state[i] = SpinupState.Delay
            else:
                out_state[i] = SpinupState.End
        elif state == SpinupState.GrowToFinalAge:
            if age[i] < final_age[i]:
                out_state[i] = SpinupState.GrowToFinalAge
            else:
                if delay[i] > 0:
                    out_state[i] = SpinupState.Delay
                else:
                    out_state[i] = SpinupState.End
    return out_state
