from typing import TYPE_CHECKING
from typing import Union
import numpy as np

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_land_state
from libcbm.model.cbm_exn import cbm_exn_annual_process_dynamics
from libcbm.model.cbm_exn import cbm_exn_disturbance_dynamics
from libcbm.model.cbm_exn import cbm_exn_growth_functions


def get_default_disturbance_ops(model: "CBMEXNModel") -> list[dict]:
    return [
        {
            "name": "disturbance",
            "op_process_name": "disturbance",
            "op_data": cbm_exn_disturbance_dynamics.disturbance(
                model.pool_names,
                model.parameters.get_disturbance_matrices(),
                model.parameters.get_disturbance_matrix_associations(),
            ),
            "requires_reindexing": True,
        }
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
    if ops is None:
        ops = get_default_disturbance_ops(model)
    for op_def in ops:
        model.matrix_ops.create_operation(**op_def)
    if op_sequence is None:
        op_sequence = get_default_disturbance_op_sequence()
    model.compute(cbm_vars, op_sequence)
    return cbm_vars


def get_default_annual_process_ops(
    model: "CBMEXNModel", cbm_vars: ModelVariables
) -> list[dict]:
    growth_info = cbm_exn_growth_functions.prepare_spinup_growth_info(
        cbm_vars,
        model.parameters.get_turnover_parameters(),
        model.parameters.get_root_parameters(),
    )
    mean_annual_temp = np.unique(
        cbm_vars["parameters"]["mean_annual_temperature"].to_numpy()
    )
    return [
        {
            "name": "snag_turnover",
            "op_process_name": "growth",
            "op_data": cbm_exn_annual_process_dynamics.snag_turnover(
                model.parameters.get_turnover_parameters(), True
            ),
            "requires_reindexing": False,
        },
        {
            "name": "biomass_turnover",
            "op_process_name": "growth",
            "op_data": cbm_exn_annual_process_dynamics.biomass_turnover(
                model.parameters.get_turnover_parameters(), True
            ),
            "requires_reindexing": False,
        },
        {
            "name": "dom_decay",
            "op_process_name": "decay",
            "op_data": cbm_exn_annual_process_dynamics.dom_decay(
                mean_annual_temp,
                model.parameters.get_decay_parameters(),
            ),
            "requires_reindexing": True,
        },
        {
            "name": "slow_decay",
            "op_process_name": "decay",
            "op_data": cbm_exn_annual_process_dynamics.slow_decay(
                mean_annual_temp,
                model.parameters.get_decay_parameters(),
            ),
            "requires_reindexing": True,
        },
        {
            "name": "slow_mixing",
            "op_process_name": "decay",
            "op_data": cbm_exn_annual_process_dynamics.slow_mixing(
                model.parameters.get_slow_mixing_rate(),
            ),
            "requires_reindexing": False,
        },
        {
            "name": "growth",
            "op_process_name": "growth",
            "op_data": cbm_exn_annual_process_dynamics.net_growth(
                growth_info,
            ),
            "requires_reindexing": True,
        },
        {
            "name": "overmature_decline",
            "op_process_name": "growth",
            "op_data": cbm_exn_annual_process_dynamics.net_growth(
                growth_info,
            ),
            "requires_reindexing": True,
        },
    ]


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
    if ops is None:
        ops = get_default_annual_process_ops(model, cbm_vars)
    for op_def in ops:
        model.matrix_ops.create_operation(**op_def)
    if op_sequence is None:
        op_sequence = get_default_annual_process_op_sequence()
    model.compute(cbm_vars, op_sequence)
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
