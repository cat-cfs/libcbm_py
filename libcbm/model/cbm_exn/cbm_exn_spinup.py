from __future__ import annotations
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union
import numpy as np

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel


from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.cbm_exn import cbm_exn_land_state
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn import cbm_exn_annual_process_dynamics
from libcbm.model.cbm_exn import cbm_exn_disturbance_dynamics
from libcbm.model.cbm_exn import cbm_exn_growth_functions


def _prepare_spinup_vars(
    spinup_input: ModelVariables,
    pool_names: list[str],
    flux_names: list[str],
    include_flux: bool,
    parameters: CBMEXNParameters,
) -> ModelVariables:
    """Initialize spinup variables and state.

    Args:
        spinup_input (ModelVariables): A collection of dataframes for running
            CBM spinup.
        pool_names (list[str]): The CBM pool names.
        flux_names (list[str]): the list of flux indicator names.  This
            parameter is ignore unless the `include_flux` parameter is set to
            True.
        include_flux (bool): If set to true space will be allocated for storing
            flux indicators through the spinup procedure.
        parameters (CBMEXNParameters): initialized CBM parameters

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
            pool_names,
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
            flux_names,
            spinup_input["parameters"].backend_type,
        )

    return ModelVariables(data)


def get_default_ops(
    model: "CBMEXNModel", spinup_vars: ModelVariables
) -> list[dict]:
    growth_info = cbm_exn_growth_functions.prepare_spinup_growth_info(
        spinup_vars,
        model.parameters.get_turnover_parameters(),
        model.parameters.get_root_parameters(),
    )
    mean_annual_temp = np.unique(
        spinup_vars["parameters"]["mean_annual_temperature"].to_numpy()
    )
    ops = [
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
            "requires_reindexing": False,
        },
        {
            "name": "slow_decay",
            "op_process_name": "decay",
            "op_data": cbm_exn_annual_process_dynamics.slow_decay(
                mean_annual_temp,
                model.parameters.get_decay_parameters(),
            ),
            "requires_reindexing": False,
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
            "name": "disturbance",
            "op_process_name": "disturbance",
            "op_data": cbm_exn_disturbance_dynamics.disturbance(
                model.pool_names,
                model.parameters.get_disturbance_matrices(),
                model.parameters.get_disturbance_matrix_associations(),
            ),
            "requires_reindexing": True,
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
    input: ModelVariables,
    reporting_func: Union[Callable[[int, ModelVariables], None], None] = None,
    include_flux: bool = False,
    ops: Union[list[dict], None] = None,
    op_sequence: Union[list[str], None] = None,
) -> ModelVariables:
    """Run the CBM spinup routine.

    Args:
        model (CBMEXNModel): Initialized cbm_exn model.
        input (ModelVariables): Spinup input, which is a collection of
            dataframes consisting of:

                * `increments`: a table of aboveground merchantable, foliage,
                    and other C increments by age by stand
                * `parameters`: CBM spinup parameters

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

    spinup_vars = _prepare_spinup_vars(
        input,
        model.pool_names,
        model.flux_names,
        include_flux,
        model.parameters,
    )
    if ops is None:
        ops = get_default_ops(model, spinup_vars)
    for op_def in ops:
        model.matrix_ops.create_operation(**op_def)
    if op_sequence is None:
        op_sequence = get_default_op_list()

    t: int = 0
    while True:
        if include_flux:
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
