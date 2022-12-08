from libcbm.model.model_definition.spinup_engine import SpinupState
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition import spinup_engine
from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
import numpy as np
import numba
from libcbm.storage import series


@numba.njit()
def _update_spinup_vars(
    n_stands: int,
    spinup_state: np.ndarray,
    out_spinup_state: np.ndarray,
    disturbance_type: np.ndarray,
    slow_c: np.ndarray,
    this_rotation_slow: np.ndarray,
    last_rotation_slow: np.ndarray,
    rotation_num: np.ndarray,
    historical_dist_type: np.ndarray,
    last_pass_dist_type: np.ndarray,
    enabled: np.ndarray,
) -> bool:
    enabled_count = n_stands
    for i in range(n_stands):
        state = spinup_state[i]
        out_spinup_state[i] = state
        if state == SpinupState.LastPassEvent.value:
            disturbance_type[i] = last_pass_dist_type[i]
        elif state == SpinupState.HistoricalEvent.value:
            disturbance_type[i] = historical_dist_type[i]
            last_rotation_slow[i] = slow_c[i]
            rotation_num[i] += 1
        else:
            if state == SpinupState.End.value:
                enabled[i] = 0
                enabled_count -= 1
            disturbance_type[i] = 0
            this_rotation_slow[i] = slow_c[i]
    return enabled_count == 0


def advance_spinup_state(
    spinup_vars: CBMVariables,
) -> tuple[bool, CBMVariables]:
    n_stands = spinup_vars["state"]["spinup_state"].length

    spinup_state = spinup_engine.advance_spinup_state(
        spinup_state=spinup_vars["state"]["spinup_state"],
        age=spinup_vars["state"]["age"] + 1,
        delay_step=spinup_vars["state"]["delay_step"],
        final_age=spinup_vars["parameters"]["age"],
        delay=spinup_vars["parameters"]["delay"],
        return_interval=spinup_vars["parameters"]["return_interval"],
        rotation_num=spinup_vars["state"]["rotation"],
        min_rotations=spinup_vars["parameters"]["min_rotations"],
        max_rotations=spinup_vars["parameters"]["max_rotations"],
        last_rotation_slow=spinup_vars["state"]["last_rotation_slow"],
        this_rotation_slow=spinup_vars["state"]["this_rotation_slow"],
        enabled=spinup_vars["state"]["enabled"],
    )

    all_finished = _update_spinup_vars(
        n_stands=n_stands,
        spinup_state=spinup_state,
        out_spinup_state=spinup_vars["state"]["spinup_state"].to_numpy(),
        disturbance_type=spinup_vars["state"]["disturbance_type"].to_numpy(),
        slow_c=(
            spinup_vars["pools"]["AboveGroundSlowSoil"].to_numpy()
            + spinup_vars["pools"]["BelowGroundSlowSoil"].to_numpy()
        ),
        this_rotation_slow=spinup_vars["state"][
            "this_rotation_slow"
        ].to_numpy(),
        last_rotation_slow=spinup_vars["state"][
            "last_rotation_slow"
        ].to_numpy(),
        rotation_num=spinup_vars["state"]["rotation"].to_numpy(),
        historical_dist_type=spinup_vars["parameters"][
            "historical_disturbance_type"
        ].to_numpy(),
        last_pass_dist_type=spinup_vars["parameters"][
            "last_pass_disturbance_type"
        ].to_numpy(),
        enabled=spinup_vars["state"]["enabled"].to_numpy(),
    )

    return all_finished, spinup_vars


def init_cbm_vars(model: CBMModel, spinup_vars: CBMVariables) -> CBMVariables:
    cbm_vars = cbm_exn_variables.init_cbm_vars(
        spinup_vars["pools"].n_rows,
        model.pool_names,
        model.flux_names,
        spinup_vars["pools"].backend_type,
    )
    for p in model.pool_names:
        cbm_vars["pools"][p].assign(spinup_vars["pools"][p])

    cbm_vars["state"]["area"].assign(spinup_vars["parameters"]["area"])
    cbm_vars["state"]["spatial_unit_id"].assign(
        spinup_vars["parameters"]["spatial_unit_id"]
    )

    cbm_vars["state"]["age"].assign(spinup_vars["parameters"]["age"])
    cbm_vars["state"]["species"].assign(spinup_vars["parameters"]["species"])
    cbm_vars["state"]["sw_hw"].assign(spinup_vars["parameters"]["sw_hw"])
    cbm_vars["state"]["time_since_last_disturbance"].assign(
        spinup_vars["parameters"]["age"] + spinup_vars["parameters"]["delay"]
    )

    # TODO implement land use change routines for the following 3 variables:
    cbm_vars["state"]["time_since_land_use_change"].assign(-1)
    cbm_vars["state"]["land_class_id"].assign(-1)
    # TODO take the enabled value from the state, and secondly assign it based
    # on deforestation/afforestation status if necessary
    cbm_vars["state"]["enabled"].assign(1)  # spinup_vars["state"]["enabled"])

    cbm_vars["state"]["last_disturbance_type"].assign(
        spinup_vars["parameters"]["last_pass_disturbance_type"]
    )

    return cbm_vars


@numba.njit()
def _end_spinup_step(
    spinup_state: np.ndarray,
    disturbance_type: np.ndarray,
    merch: np.ndarray,
    foliage: np.ndarray,
    other: np.ndarray,
    fine_root: np.ndarray,
    coarse_root: np.ndarray,
    age: np.ndarray,
    delay_step: np.ndarray,
):
    n_rows = spinup_state.shape[0]
    for i in range(n_rows):
        if spinup_state[i] == SpinupState.End:
            break
        age[i] += 1

        if disturbance_type[i] > 0:
            merch[i] = 0
            foliage[i] = 0
            other[i] = 0
            fine_root[i] = 0
            coarse_root[i] = 0
            age[i] = 0

        if spinup_state[i] == SpinupState.Delay.value:
            delay_step[i] += 1
            age[i] = 0


def end_spinup_step(spinup_vars: CBMVariables) -> CBMVariables:

    _end_spinup_step(
        spinup_state=spinup_vars["state"]["spinup_state"].to_numpy(),
        disturbance_type=spinup_vars["state"]["disturbance_type"].to_numpy(),
        merch=spinup_vars["pools"]["Merch"].to_numpy(),
        foliage=spinup_vars["pools"]["Foliage"].to_numpy(),
        other=spinup_vars["pools"]["Other"].to_numpy(),
        fine_root=spinup_vars["pools"]["FineRoots"].to_numpy(),
        coarse_root=spinup_vars["pools"]["CoarseRoots"].to_numpy(),
        age=spinup_vars["state"]["age"].to_numpy(),
        delay_step=spinup_vars["state"]["delay_step"].to_numpy(),
    )

    return spinup_vars


def start_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    idx = series.from_numpy("", np.arange(0, cbm_vars["pools"].n_rows))
    disturbed_idx = idx.filter(cbm_vars["parameters"]["disturbance_type"] > 0)

    # currently only considering age-resetting disturbances
    cbm_vars["state"]["age"].assign(0, disturbed_idx)

    cbm_vars["state"]["last_disturbance_type"].assign(
        cbm_vars["parameters"]["disturbance_type"].take(disturbed_idx),
        disturbed_idx,
    )
    cbm_vars["state"]["time_since_last_disturbance"].assign(0, disturbed_idx)

    # TODO implement land use change routines for the following 3 variables:
    cbm_vars["state"]["time_since_land_use_change"].assign(-1)
    cbm_vars["state"]["land_class_id"].assign(-1)
    # TODO assign enabled/disturbance based
    # on deforestation/afforestation status if necessary
    # cbm_vars["state"]["enabled"].assign(...)
    return cbm_vars


def end_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    idx = series.from_numpy("", np.arange(0, cbm_vars["pools"].n_rows))
    enabled_idx = idx.filter(cbm_vars["state"]["enabled"] > 0)
    cbm_vars["state"]["age"].assign(
        cbm_vars["state"]["age"].take(enabled_idx) + 1, enabled_idx
    )
    cbm_vars["state"]["time_since_last_disturbance"].assign(
        cbm_vars["state"]["time_since_last_disturbance"].take(enabled_idx) + 1,
        enabled_idx,
    )

    # TODO increment time_since_land_use_change where the values are >= 0
    # cbm_vars["state"]["time_since_land_use_change"].assign(-1)
    return cbm_vars
