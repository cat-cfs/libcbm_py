from libcbm.model.model_definition.spinup_engine import SpinupState
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition import spinup_engine
from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
import numpy as np
from libcbm.storage import series


def advance_spinup_state(spinup_vars: CBMVariables) -> CBMVariables:
    # TODO, validate this input
    spinup_state = spinup_engine.advance_spinup_state(
        spinup_state=spinup_vars["state"]["spinup_state"],
        age=spinup_vars["state"]["age"],
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
    spinup_vars["state"]["spinup_state"].assign(spinup_state)


def init_cbm_vars(model: CBMModel, spinup_vars: CBMVariables) -> CBMVariables:
    cbm_vars = cbm_exn_variables.init_cbm_vars(
        spinup_vars["pools"].n_rows,
        model.pool_names,
        model.flux_names,
        spinup_vars["pools"].backend_type,
    )
    for p in model.pool_names:
        cbm_vars["pools"][p].assign(spinup_vars["pools"][p])

    cbm_vars["state"]["area"].assign(spinup_vars["state"]["area"])
    cbm_vars["state"]["spatial_unit_id"].assign(spinup_vars["state"]["area"])

    cbm_vars["state"]["age"].assign(spinup_vars["state"]["age"])
    cbm_vars["state"]["species"].assign(spinup_vars["state"]["species"])
    cbm_vars["state"]["time_since_last_disturbance"].assign(
        spinup_vars["state"]["age"] + spinup_vars["state"]["delay"]
    )

    # TODO implement land use change routines for the following 3 variables:
    cbm_vars["state"]["time_since_land_use_change"].assign(-1)
    cbm_vars["state"]["land_class_d"].assign(-1)
    # TODO take the enabled value from the state, and secondly assign it based
    # on deforestation/afforestation status if necessary
    cbm_vars["state"]["enabled"].assign(spinup_vars["state"]["enabled"])

    cbm_vars["state"]["last_disturbance_type_id"].assign(
        spinup_vars["spinup_parameters"]["last_pass_disturbance_type_id"]
    )

    return cbm_vars


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


def start_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    idx = series.from_numpy("", np.arange(0, cbm_vars["pools"].n_rows))
    disturbed_idx = idx.filter(cbm_vars["state"]["disturbance_type"] > 0)

    # currently only considering age-resetting disturbances
    cbm_vars["state"]["age"].assign(0, disturbed_idx)

    cbm_vars["state"]["last_disturbance_type"].assign(
        cbm_vars["state"]["disturbance_type"].take(disturbed_idx),
        disturbed_idx
    )
    cbm_vars["state"]["time_since_last_disturbance"].assign(0, disturbed_idx)

    # TODO implement land use change routines for the following 3 variables:
    cbm_vars["state"]["time_since_land_use_change"].assign(-1)
    cbm_vars["state"]["land_class_d"].assign(-1)
    # TODO assign enabled/disturbance based
    # on deforestation/afforestation status if necessary
    # cbm_vars["state"]["enabled"].assign(...)


def end_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    idx = series.from_numpy("", np.arange(0, cbm_vars["pools"].n_rows))
    enabled_idx = idx.filter(cbm_vars["state"]["enabled"] > 0)
    cbm_vars["state"]["age"].assign(
        cbm_vars["state"]["age"].take(enabled_idx) + 1,
        enabled_idx
    )
    cbm_vars["state"]["time_since_last_disturbance"].assign(
        cbm_vars["state"]["time_since_last_disturbance"].take(enabled_idx) + 1,
        enabled_idx
    )

    # TODO increment time_since_land_use_change where the values are >= 0
    # cbm_vars["state"]["time_since_land_use_change"].assign(-1)
