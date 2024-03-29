# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from typing import Callable
from libcbm.storage.dataframe import DataFrame
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.cbm_model import CBM
from libcbm.storage.backends import BackendType


def simulate(
    cbm: CBM,
    n_steps: int,
    classifiers: DataFrame,
    inventory: DataFrame,
    reporting_func: Callable[[int, CBMVariables], None],
    pre_dynamics_func: Callable[[int, CBMVariables], CBMVariables] = None,
    spinup_params: DataFrame = None,
    spinup_reporting_func: Callable[[int, CBMVariables], None] = None,
    backend_type: BackendType = None,
):
    """Runs the specified number of timesteps of the CBM model.  Model output
    is processed by the provided reporting_func. The provided
    pre_dynamics_func is called prior to each CBM dynamics step.

    Args:
        cbm (libcbm.model.cbm.cbm_model.CBM): Instance of the CBM model
        n_steps (int): The number of CBM timesteps to run
        classifiers (DataFrame): CBM classifiers for each of the rows
            in the inventory
        inventory (DataFrame): CBM inventory which defines the initial
            state of the simulation
        reporting_func (function): a function which accepts the simulation
            timestep and all CBM variables for reporting results by timestep.
        pre_dynamics_func (function, optional): A function which accepts the
            simulation timestep and all CBM variables, and which is called
            prior to computing C dynamics. The function returns all CBM
            variables which will then be passed into the current CBM timestep.
        spinup_params (object): Collection of spinup specific parameters.
            If unspecified, CBM default values are used. See
            :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_parameters`
            for object format
        spinup_reporting_func (function, optional): a function which accepts
            the spinup iteration, and all spinup variables.  Specifying this
            function will result in a performance penalty as the per-iteration
            spinup results are computed and tracked. If unspecified spinup
            results are not tracked. Defaults to None.
        backend_type (BackendType): specifies the backend storage method for
            dataframes. If unspecified, the inventory data frame's backend
            type is used.
    """
    if not backend_type:
        backend_type = inventory.backend_type
    cbm_vars = cbm_variables.initialize_simulation_variables(
        classifiers,
        inventory,
        cbm.pool_codes,
        cbm.flux_indicator_codes,
        backend_type,
    )

    spinup_vars = cbm_variables.initialize_spinup_variables(
        cbm_vars,
        backend_type,
        spinup_params,
        include_flux=spinup_reporting_func is not None,
    )

    cbm.spinup(spinup_vars, reporting_func=spinup_reporting_func)

    if "mean_annual_temp" in spinup_vars.parameters.columns:
        # since the mean_annual_temp appears in the spinup parameters, carry
        # it forward to the simulation period so that we have consistent
        # columns in the outputs
        cbm_vars.parameters.add_column(
            spinup_vars.parameters["mean_annual_temp"],
            cbm_vars.parameters.n_cols,
        )
    cbm_vars = cbm.init(cbm_vars)
    reporting_func(0, cbm_vars)

    for time_step in range(1, int(n_steps) + 1):
        if pre_dynamics_func:
            cbm_vars = pre_dynamics_func(time_step, cbm_vars)

        cbm_vars = cbm.step(cbm_vars)
        reporting_func(time_step, cbm_vars)
