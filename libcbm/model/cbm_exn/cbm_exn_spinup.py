from typing import Callable
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.cbm_exn import cbm_exn_matrix_ops
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn import cbm_exn_land_state


def prepare_spinup_vars(
    spinup_input: CBMVariables, include_flux: bool
) -> CBMVariables:
    pass


def spinup(
    model: CBMModel,
    input: CBMVariables,
    reporting_func: Callable[[int, CBMVariables], None] = None,
    include_flux: bool = False,
) -> CBMVariables:

    parameters = CBMEXNParameters(model.parameters)
    matrix_ops = cbm_exn_matrix_ops.SpinupMatrixOps(model, parameters)

    spinup_vars = prepare_spinup_vars(input, include_flux)
    n_stands = spinup_vars["pools"].n_rows
    t: int = 0
    while True:
        n_finished = cbm_exn_land_state.advance_spinup_state(spinup_vars)
        if n_finished == n_stands:
            break

        step_operations = matrix_ops.get_spinup_matrix_ops(spinup_vars)
        ops = [
            step_operations["growth"],
            step_operations["snag_turnover"],
            step_operations["biomass_turnover"],
            step_operations["overmature_decline"],
            step_operations["growth"],
            step_operations["dom_decay"],
            step_operations["slow_decay"],
            step_operations["slow_mixing"],
        ]
        op_process_ids = (
            [cbm_exn_matrix_ops.OpProcesses.disturbance]
            + [cbm_exn_matrix_ops.OpProcesses.growth] * 5
            + [cbm_exn_matrix_ops.OpProcesses.decay] * 3
        )
        model.compute(spinup_vars, ops, op_process_ids)
        spinup_vars = cbm_exn_land_state.end_spinup_step(spinup_vars)
        t += 1
        if reporting_func:
            reporting_func(t, spinup_vars)

    return cbm_exn_land_state.init_cbm_vars(spinup_vars)
