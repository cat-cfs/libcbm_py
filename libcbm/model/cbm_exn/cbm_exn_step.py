from libcbm.model.cbm_exn.model import CBMEXNModel
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn import cbm_exn_matrix_ops
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


def step_disturbance(
    model: CBMEXNModel,
    cbm_vars: CBMVariables,
    parameters: CBMEXNParameters = None,
) -> CBMVariables:
    if not parameters:
        parameters = CBMEXNParameters(model.parameters)
    disturbance = cbm_exn_matrix_ops.disturbance(model, cbm_vars, parameters)
    model.compute(
        cbm_vars, [disturbance], [cbm_exn_matrix_ops.OpProcesses.disturbance]
    )


def step_annual_process(
    model: CBMEXNModel,
    cbm_vars: CBMVariables,
    parameters: CBMEXNParameters = None,
) -> CBMVariables:
    if not parameters:
        parameters = CBMEXNParameters(model.parameters)
    growth_op = cbm_exn_matrix_ops.net_growth(model, cbm_vars, parameters)
    ops = [
        growth_op,
        cbm_exn_matrix_ops.snag_turnover(model, cbm_vars, parameters),
        cbm_exn_matrix_ops.biomass_turnover(model, cbm_vars, parameters),
        cbm_exn_matrix_ops.overmature_decline(model, cbm_vars, parameters),
        growth_op,
        cbm_exn_matrix_ops.dom_decay(model, cbm_vars, parameters),
        cbm_exn_matrix_ops.slow_decay(model, cbm_vars, parameters),
        cbm_exn_matrix_ops.slow_mixing(model, cbm_vars, parameters),
    ]
    op_process_ids = [cbm_exn_matrix_ops.OpProcesses.growth] * 5 + [
        cbm_exn_matrix_ops.OpProcesses.decay
    ] * 3
    model.compute(cbm_vars, ops, op_process_ids)
    for op in ops:
        op.dispose()
    return cbm_vars


def start_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    pass


def end_step(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> CBMVariables:
    pass


def step(model: CBMEXNModel, cbm_vars: CBMVariables) -> CBMVariables:
    parameters = CBMEXNParameters(model.parameters)
    cbm_vars = start_step(cbm_vars)
    cbm_vars = step_disturbance(model, cbm_vars, parameters)
    cbm_vars = step_annual_process(model, cbm_vars, parameters)
    cbm_vars = end_step(cbm_vars)
    return cbm_vars
