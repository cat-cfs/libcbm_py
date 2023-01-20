from __future__ import annotations
from typing import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel

from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_land_state
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


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


def spinup(
    model: "CBMEXNModel",
    input: ModelVariables,
    reporting_func: Callable[[int, ModelVariables], None] = None,
    include_flux: bool = False,
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

    snag_turnover = model.matrix_ops.snag_turnover(
        spinup_vars["parameters"]["spatial_unit_id"],
        spinup_vars["parameters"]["sw_hw"],
    )
    biomass_turnover = model.matrix_ops.biomass_turnover(
        spinup_vars["parameters"]["spatial_unit_id"],
        spinup_vars["parameters"]["sw_hw"],
    )
    dom_decay = model.matrix_ops.dom_decay(
        spinup_vars["parameters"]["mean_annual_temperature"]
    )
    slow_decay = model.matrix_ops.slow_decay(
        spinup_vars["parameters"]["mean_annual_temperature"]
    )
    slow_mixing = model.matrix_ops.slow_mixing(
        spinup_vars["parameters"].n_rows
    )
    t: int = 0
    while True:

        if include_flux:
            spinup_vars["flux"].zero()
        all_finished, spinup_vars = cbm_exn_land_state.advance_spinup_state(
            spinup_vars
        )
        if all_finished:
            break

        disturbance = model.matrix_ops.disturbance(
            spinup_vars["state"]["disturbance_type"],
            spinup_vars["parameters"]["spatial_unit_id"],
            spinup_vars["parameters"]["sw_hw"],
        )
        growth, overmature_decline = model.matrix_ops.spinup_net_growth(
            spinup_vars
        )

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
        model.compute(spinup_vars, ops)
        spinup_vars = cbm_exn_land_state.end_spinup_step(spinup_vars)
        if reporting_func:
            reporting_func(t, spinup_vars)
        t += 1

    return cbm_exn_land_state.init_cbm_vars(model, spinup_vars)
