from libcbm.model.model_definition.spinup_engine import SpinupState
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition import spinup_engine
import numpy as np
from libcbm.storage import series


def advance_spinup_state(spinup_vars: CBMVariables) -> CBMVariables:
    spinup_state = spinup_engine.advance_spinup_state(
        spinup_state=spinup_vars["state"]["spinup_state"],
        age=spinup_vars["state"]["age"],
        final_age=spinup_vars["parameters"]["age"],
        final_age=spinup_vars["parameters"]["age"],
        return_interval=spinup_vars["parameters"]["return_interval"],
        rotation_num=spinup_vars["state"]["rotation"],
        max_rotations=spinup_vars["parameters"]["max_rotations"],
        min_rotations=spinup_vars["parameters"]["min_rotations"],
        this_rotation_slow=spinup_vars["state"]["this_rotation_slow"],
        last_rotation_slow=spinup_vars["state"]["last_rotation_slow"],
    )
    spinup_vars["state"]["spinup_state"].assign(spinup_state)


def init_cbm_vars(model: CBMModel, spinup_vars: CBMVariables) -> CBMVariables:
    pass


def end_spinup_step(spinup_vars: CBMVariables) -> CBMVariables:
    idx = series.from_numpy("", np.arange(0, spinup_vars["pools"].n_rows))
    disturbed_idx = idx.filter(spinup_vars["state"]["disturbance_type"] > 0)

    growing_idx = idx.filter(
        (spinup_vars["state"]["spinup_state"] == SpinupState.GrowToFinalAge)
        | (spinup_vars["state"]["spinup_state"] == SpinupState.AnnualProcesses)
    )
    delay_idx = idx.filter(
        spinup_vars["state"]["spinup_state"] == SpinupState.Delay
    )

    spinup_vars["pools"]["Merch"].assign(0, disturbed_idx)
    spinup_vars["pools"]["Foliage"].assign(0, disturbed_idx)
    spinup_vars["pools"]["Other"].assign(0, disturbed_idx)
    spinup_vars["pools"]["FineRoots"].assign(0, disturbed_idx)
    spinup_vars["pools"]["CoarseRoots"].assign(0, disturbed_idx)

    spinup_vars["state"]["last_rotation_slow_c"] = (
        spinup_vars["pools"]["AboveGroundSlow"]
        + spinup_vars["pools"]["BelowGroundSlow"]
    )

    spinup_vars["state"]["age"].assign(
        spinup_vars["state"]["age"].take(growing_idx) + 1, growing_idx
    )

    spinup_vars["state"]["delay_step"].assign(
        spinup_vars["state"]["delay_step"].take(delay_idx) + 1, delay_idx
    )
