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
    matrix_ops = cbm_exn_matrix_ops.MatrixOps(model, parameters)

    spinup_vars = prepare_spinup_vars(input, include_flux)
    n_stands = spinup_vars["pools"].n_rows
    snag_turnover = matrix_ops.snag_turnover(
        spinup_vars["state"]["spatial_unit_id"]
    )
    biomass_turnover = matrix_ops.biomass_turnover(
        spinup_vars["state"]["spatial_unit_id"]
    )
    dom_decay = matrix_ops.dom_decay(
        spinup_vars["parameters"]["mean_annual_temperature"]
    )
    slow_decay = matrix_ops.slow_decay(
        spinup_vars["parameters"]["mean_annual_temperature"]
    )
    slow_mixing = matrix_ops.slow_mixing(spinup_vars["parameters"].n_rows)
    t: int = 0
    while True:
        n_finished = cbm_exn_land_state.advance_spinup_state(spinup_vars)
        if n_finished == n_stands:
            break

        disturbance = matrix_ops.disturbance(
            spinup_vars["state"]["disturbance_type"],
            spinup_vars["state"]["spatial_unit_id"],
        )
        growth, overmature_decline = matrix_ops.spinup_net_growth(spinup_vars)
        matrix_ops.net_growth()
        ops = [
            growth,
            snag_turnover,
            biomass_turnover,
            overmature_decline,
            growth,
            dom_decay,
            slow_decay,
            slow_mixing,
            disturbance,
        ]
        op_process_ids = (
            [cbm_exn_matrix_ops.OpProcesses.growth] * 5
            + [cbm_exn_matrix_ops.OpProcesses.decay] * 3
            + [cbm_exn_matrix_ops.OpProcesses.disturbance]
        )
        model.compute(spinup_vars, ops, op_process_ids)
        spinup_vars = cbm_exn_land_state.end_spinup_step(spinup_vars)
        t += 1
        if reporting_func:
            reporting_func(t, spinup_vars)

    return cbm_exn_land_state.init_cbm_vars(spinup_vars)
