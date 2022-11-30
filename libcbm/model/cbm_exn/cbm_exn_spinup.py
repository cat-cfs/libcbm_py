from typing import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libcbm.model.cbm_exn.cbm_exn_model import CBMEXNModel

from libcbm.model.cbm_exn import cbm_exn_variables
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.cbm_exn import cbm_exn_land_state


def prepare_spinup_vars(
    spinup_input: CBMVariables,
    pool_names: list[str],
    flux_names: list[str],
    include_flux: bool,
) -> CBMVariables:
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
    data["pools"]["Input"].assign(1.0)
    if include_flux:
        data["flux"] = cbm_exn_variables.init_flux(
            spinup_input["parameters"].n_rows,
            flux_names,
            spinup_input["parameters"].backend_type,
        )

    return CBMVariables(data)


def spinup(
    model: "CBMEXNModel",
    input: CBMVariables,
    reporting_func: Callable[[int, CBMVariables], None] = None,
    include_flux: bool = False,
) -> CBMVariables:

    spinup_vars = prepare_spinup_vars(
        input, model.pool_names, model.flux_names, include_flux
    )

    snag_turnover = model.matrix_ops.snag_turnover(
        spinup_vars["parameters"]["spatial_unit_id"],
        spinup_vars["parameters"]["species"],
    )
    biomass_turnover = model.matrix_ops.biomass_turnover(
        spinup_vars["parameters"]["spatial_unit_id"],
        spinup_vars["parameters"]["species"],
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
        all_finished, spinup_vars = cbm_exn_land_state.advance_spinup_state(
            spinup_vars
        )
        if all_finished:
            break

        disturbance = model.matrix_ops.disturbance(
            spinup_vars["state"]["disturbance_type"],
            spinup_vars["parameters"]["spatial_unit_id"],
            spinup_vars["parameters"]["species"],
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
        t += 1
        if reporting_func:
            reporting_func(t, spinup_vars)

    return cbm_exn_land_state.init_cbm_vars(model, spinup_vars)
