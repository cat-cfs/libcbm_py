from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_land_state
from libcbm.model.cbm_exn import cbm_exn_annual_process_dynamics
from libcbm.model.cbm_exn import cbm_exn_disturbance_dynamics
from libcbm.model.cbm_exn import cbm_exn_growth_functions


def get_default_ops(
    parameters: CBMEXNParameters, cbm_vars: ModelVariables, which: str = "all"
) -> list[dict]:
    if which not in ["disturbance", "annual_process", "all"]:
        raise ValueError(f"uknown parameter value for which '{which}'")

    output = []

    if which in ["disturbance", "all"]:
        output.append(
            {
                "name": "disturbance",
                "op_process_name": "Disturbance",
                "op_data": cbm_exn_disturbance_dynamics.disturbance(
                    parameters.pool_configuration(),
                    parameters.get_disturbance_matrices(),
                    parameters.get_disturbance_matrix_associations(),
                    False,
                ),
                "requires_reindexing": True,
            }
        )

    if which in ["annual_process", "all"]:
        growth_info = cbm_exn_growth_functions.prepare_growth_info(
            cbm_vars,
            parameters.get_turnover_parameters(),
            parameters.get_root_parameters(),
        )
        annual_process_ops = [
            {
                "name": "snag_turnover",
                "op_process_name": "Growth and Turnover",
                "op_data": cbm_exn_annual_process_dynamics.snag_turnover(
                    parameters.get_turnover_parameters(), False
                ),
                "requires_reindexing": False,
            },
            {
                "name": "biomass_turnover",
                "op_process_name": "Growth and Turnover",
                "op_data": cbm_exn_annual_process_dynamics.biomass_turnover(
                    parameters.get_turnover_parameters(), False
                ),
                "requires_reindexing": False,
            },
            {
                "name": "dom_decay",
                "op_process_name": "Decay",
                "op_data": cbm_exn_annual_process_dynamics.dom_decay(
                    cbm_vars["parameters"][
                        "mean_annual_temperature"
                    ].to_numpy(),
                    parameters.get_decay_parameters(),
                ),
                "requires_reindexing": True,
            },
            {
                "name": "slow_decay",
                "op_process_name": "Decay",
                "op_data": cbm_exn_annual_process_dynamics.slow_decay(
                    cbm_vars["parameters"][
                        "mean_annual_temperature"
                    ].to_numpy(),
                    parameters.get_decay_parameters(),
                ),
                "requires_reindexing": True,
            },
            {
                "name": "slow_mixing",
                "op_process_name": "Decay",
                "op_data": cbm_exn_annual_process_dynamics.slow_mixing(
                    parameters.get_slow_mixing_rate(),
                ),
                "requires_reindexing": False,
            },
            {
                "name": "growth",
                "op_process_name": "Growth and Turnover",
                "op_data": cbm_exn_annual_process_dynamics.net_growth(
                    growth_info,
                ),
                "requires_reindexing": True,
            },
            {
                "name": "overmature_decline",
                "op_process_name": "Growth and Turnover",
                "op_data": cbm_exn_annual_process_dynamics.overmature_decline(
                    growth_info,
                ),
                "requires_reindexing": True,
            },
        ]
        output.extend(annual_process_ops)

    return output


def get_default_annual_process_op_sequence() -> list[str]:
    return [
        "growth",
        "snag_turnover",
        "biomass_turnover",
        "overmature_decline",
        "growth",
        "dom_decay",
        "slow_decay",
        "slow_mixing",
    ]


def get_default_disturbance_op_sequence() -> list[str]:
    return ["disturbance"]


def step_disturbance(
    model: "CBMEXNModel",
    cbm_vars: ModelVariables,
    ops: Union[list[dict], None] = None,
    op_sequence: Union[list[str], None] = None,
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
    if op_sequence is None:
        op_sequence = get_default_disturbance_op_sequence()
    if ops is None:
        ops = get_default_ops(model.parameters, cbm_vars, "disturbance")
    for op_def in ops:
        if op_def["name"] in op_sequence:
            model.matrix_ops.create_operation(**op_def)

    model.compute(cbm_vars, op_sequence)
    return cbm_vars


def step_annual_process(
    model: "CBMEXNModel",
    cbm_vars: ModelVariables,
    ops: Union[list[dict], None] = None,
    op_sequence: Union[list[str], None] = None,
) -> ModelVariables:
    """Compute and track CBM annual processes.

    Args:
        model (CBMEXNModel): initialized cbm_exn model
        cbm_vars (CBMVariables): cbm variables and state

    Returns:
        CBMVariables: updated cbm_vars
    """
    if op_sequence is None:
        op_sequence = get_default_annual_process_op_sequence()
    if ops is None:
        ops = get_default_ops(model.parameters, cbm_vars, "annual_process")
    for op_def in ops:
        if op_def["name"] in op_sequence:
            model.matrix_ops.create_operation(**op_def)

    model.compute(cbm_vars, op_sequence)
    return cbm_vars


def step(
    model: "CBMEXNModel",
    cbm_vars: ModelVariables,
    ops: Union[list[dict], None] = None,
    step_op_sequence: Union[list[str], None] = None,
    disturbance_op_sequence: Union[list[str], None] = None,
) -> ModelVariables:
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
    if ops is None:
        ops = get_default_ops(model.parameters, cbm_vars, "all")
    for op_def in ops:
        model.matrix_ops.create_operation(**op_def)

    cbm_vars["flux"].zero()
    cbm_vars = cbm_exn_land_state.start_step(cbm_vars, model.parameters)
    cbm_vars = step_disturbance(model, cbm_vars, ops, disturbance_op_sequence)
    cbm_vars = step_annual_process(model, cbm_vars, ops, step_op_sequence)
    cbm_vars = cbm_exn_land_state.end_step(cbm_vars, model.parameters)
    return cbm_vars
