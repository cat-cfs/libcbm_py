from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_land_state


def step_disturbance(
    model: "CBMEXNModel",
    cbm_vars: ModelVariables,
) -> ModelVariables:
    """Compute and track disturbance matrix effects across multiple stands.

    The "disturbance_type" series of the `cbm_vars` `parameters` dataframe
    is used to set the disturbance type. Zero or negative values indicate no
    disturbance.

    The spatial_unit_id, and sw_hw series of the `cbm_vars` `state` dataframe
    is used to select the appropriate disturbance matrix.

    Args:
        model (CBMEXNModel): initialized cbm_exn model
        cbm_vars (ModelVariables): cbm variables and state

    Returns:
        ModelVariables: updated cbm_variables and state
    """
    disturbance = model.matrix_ops.disturbance(
        cbm_vars["parameters"]["disturbance_type"],
        cbm_vars["state"]["spatial_unit_id"],
        cbm_vars["state"]["sw_hw"],
    )
    model.compute(cbm_vars, [disturbance])
    return cbm_vars


def step_annual_process(
    model: "CBMEXNModel",
    cbm_vars: ModelVariables,
) -> ModelVariables:
    """Compute and track CBM annual processes.

    Args:
        model (CBMEXNModel): initialized cbm_exn model
        cbm_vars (CBMVariables): cbm variables and state

    Returns:
        CBMVariables: updated cbm_vars
    """
    growth_op, overmature_decline = model.matrix_ops.net_growth(cbm_vars)
    spuid = cbm_vars["state"]["spatial_unit_id"]
    sw_hw = cbm_vars["state"]["sw_hw"]
    mean_annual_temp = cbm_vars["parameters"]["mean_annual_temperature"]
    ops = [
        growth_op,
        model.matrix_ops.snag_turnover(spuid, sw_hw),
        model.matrix_ops.biomass_turnover(spuid, sw_hw),
        overmature_decline,
        growth_op,
        model.matrix_ops.dom_decay(mean_annual_temp),
        model.matrix_ops.slow_decay(mean_annual_temp),
        model.matrix_ops.slow_mixing(spuid.length),
    ]

    model.compute(cbm_vars, ops)
    for op in ops:
        op.dispose()
    return cbm_vars


def step(model: "CBMEXNModel", cbm_vars: ModelVariables) -> ModelVariables:
    """Advance CBM state by one timestep, and track results.

    This function updates state variables, performs disturbances for affected
    statnds and annaul processes for all stands.  The cbm_vars `pools`
    dataframe is updated with changed pool values. The cbm_vars `flux`
    dataframe tracks specific flows between pools for meaningful indicators.

    Args:
        model (CBMEXNModel): initialized cbm_exn model
        cbm_vars (ModelVariables): cbm variables and state

    Returns:
        ModelVariables: updated cbm_vars
    """

    cbm_vars["flux"].zero()
    cbm_vars = cbm_exn_land_state.start_step(cbm_vars, model.parameters)
    cbm_vars = step_disturbance(model, cbm_vars)
    cbm_vars = step_annual_process(model, cbm_vars)
    cbm_vars = cbm_exn_land_state.end_step(cbm_vars, model.parameters)
    return cbm_vars
