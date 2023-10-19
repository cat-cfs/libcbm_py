from __future__ import annotations
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.cbm_exn import cbm_exn_land_state
from libcbm.model.cbm_exn import cbm_exn_annual_process_dynamics
from libcbm.model.cbm_exn import cbm_exn_disturbance_dynamics
from libcbm.model.cbm_exn import cbm_exn_growth_functions


def prepare_spinup_vars(
    spinup_input: ModelVariables,
    parameters: CBMEXNParameters,
    include_flux: bool = False,
) -> ModelVariables:
    """Initialize spinup variables and state.

    Args:
        spinup_input (ModelVariables): A collection of dataframes for running
            CBM spinup.
        model (CBMEXNModel): Initialized cbm_exn model.
        include_flux (bool): If set to true space will be allocated for storing
            flux indicators through the spinup procedure.

    Returns:
        ModelVariables: Inititlized cbm variables and state for running spinup.
    """
    data = {
        "parameters": spinup_input["parameters"],
        "increments": spinup_input["increments"],
        "state": cbm_exn_variables.init_spinup_state(
            spinup_input["parameters"].n_rows,
            spinup_input["parameters"].backend_type,
        ),
        "pools": cbm_exn_variables.init_pools(
            spinup_input["parameters"].n_rows,
            parameters.pool_configuration(),
            spinup_input["parameters"].backend_type,
        ),
    }
    if "sw_hw" not in data["parameters"].columns:
        sw_hw = data["parameters"]["species"].map(parameters.get_sw_hw_map())
        sw_hw.name = "sw_hw"
        data["parameters"].add_column(sw_hw, 0)
    data["pools"]["Input"].assign(1.0)
    if include_flux:
        data["flux"] = cbm_exn_variables.init_flux(
            spinup_input["parameters"].n_rows,
            [f["name"] for f in parameters.flux_configuration()],
            spinup_input["parameters"].backend_type,
        )

    return ModelVariables(data)


def get_default_ops(
    parameters: CBMEXNParameters, spinup_vars: ModelVariables
) -> list[dict]:
    growth_info = cbm_exn_growth_functions.prepare_spinup_growth_info(
        spinup_vars,
        parameters.get_turnover_parameters(),
        parameters.get_root_parameters(),
    )
    net_growth = cbm_exn_annual_process_dynamics.net_growth(
        growth_info,
    )
    overmature_decline = cbm_exn_annual_process_dynamics.overmature_decline(
        growth_info,
    )

    ops = [
        {
            "name": "snag_turnover",
            "op_process_name": "Growth and Turnover",
            "op_data": cbm_exn_annual_process_dynamics.snag_turnover(
                parameters.get_turnover_parameters(), True
            ),
            "requires_reindexing": False,
        },
        {
            "name": "biomass_turnover",
            "op_process_name": "Growth and Turnover",
            "op_data": cbm_exn_annual_process_dynamics.biomass_turnover(
                parameters.get_turnover_parameters(), True
            ),
            "requires_reindexing": False,
        },
        {
            "name": "dom_decay",
            "op_process_name": "Decay",
            "op_data": cbm_exn_annual_process_dynamics.dom_decay(
                spinup_vars["parameters"][
                    "mean_annual_temperature"
                ].to_numpy(),
                parameters.get_decay_parameters(),
            ),
            "requires_reindexing": False,
        },
        {
            "name": "slow_decay",
            "op_process_name": "Decay",
            "op_data": cbm_exn_annual_process_dynamics.slow_decay(
                spinup_vars["parameters"][
                    "mean_annual_temperature"
                ].to_numpy(),
                parameters.get_decay_parameters(),
            ),
            "requires_reindexing": False,
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
            "name": "disturbance",
            "op_process_name": "Disturbance",
            "op_data": cbm_exn_disturbance_dynamics.disturbance(
                parameters.pool_configuration(),
                parameters.get_disturbance_matrices(),
                parameters.get_disturbance_matrix_associations(),
                True,
            ),
            "requires_reindexing": True,
        },
        {
            "name": "growth",
            "op_process_name": "Growth and Turnover",
            "op_data": net_growth,
            "requires_reindexing": True,
            "default_matrix_index": len(net_growth.index) - 1,
        },
        {
            "name": "overmature_decline",
            "op_process_name": "Growth and Turnover",
            "op_data": overmature_decline,
            "requires_reindexing": True,
            "default_matrix_index": len(overmature_decline.index) - 1,
        },
    ]
    return ops


def get_default_op_list() -> list[str]:
    return [
        "growth",
        "snag_turnover",
        "biomass_turnover",
        "overmature_decline",
        "growth",
        "dom_decay",
        "slow_decay",
        "slow_mixing",
        "disturbance",
    ]


def spinup(
    model: "CBMEXNModel",
    spinup_vars: ModelVariables,
    reporting_func: Union[Callable[[int, ModelVariables], None], None] = None,
    ops: Union[list[dict], None] = None,
    op_sequence: Union[list[str], None] = None,
) -> ModelVariables:
    """Run the CBM spinup routine.

    Args:
        model (CBMEXNModel): Initialized cbm_exn model.
        spinup_vars (ModelVariables): Spinup vars, as returned by
            :py:func:`cbm_exn_spinup.prepare_spinup_vars`.
        reporting_func (Callable[[int, ModelVariables], None], optional):
            Optional function for accepting timestep-by-timestep spinup
            results for debugging. Defaults to None.
        include_flux (bool, optional): if reporting func is specified,
            flux values will additionally be tracked during the spinup
            process. Defaults to False.

    Returns:
        ModelVariables: A collection of dataframes with initialized C pools and
            state, ready for CBM stepping.
    """

    if ops is None:
        ops = get_default_ops(model.parameters, spinup_vars)
    for op_def in ops:
        model.matrix_ops.create_operation(**op_def)
    if op_sequence is None:
        op_sequence = get_default_op_list()

    t: int = 0
    while True:
        if "flux" in spinup_vars:
            spinup_vars["flux"].zero()
        all_finished, spinup_vars = cbm_exn_land_state.advance_spinup_state(
            spinup_vars
        )
        if all_finished:
            break

        model.compute(spinup_vars, op_sequence)
        spinup_vars = cbm_exn_land_state.end_spinup_step(spinup_vars)
        if reporting_func:
            reporting_func(t, spinup_vars)
        t += 1

    return cbm_exn_land_state.init_cbm_vars(model, spinup_vars)
