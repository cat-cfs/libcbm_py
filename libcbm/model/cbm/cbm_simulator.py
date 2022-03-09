# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from typing import Callable
from libcbm.storage.dataframe import DataFrame
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.cbm_model import CBM


def simulate(
    cbm: CBM,
    n_steps: int,
    classifiers: DataFrame,
    inventory: DataFrame,
    reporting_func: Callable[[int, CBMVariables], None],
    pre_dynamics_func: Callable[[int, CBMVariables], CBMVariables] = None,
    spinup_params: DataFrame = None,
    spinup_reporting_func: Callable[[int, CBMVariables], None] = None,
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
        spinup_params (object): collection of spinup specific parameters. See
            :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_parameters`
            for object format
        spinup_reporting_func (function, optional): a function which accepts
            the spinup iteration, and all spinup variables.  Specifying this
            function will result in a performance penalty as the per-iteration
            spinup results are computed and tracked. If unspecified spinup
            results are not tracked. Defaults to None.
    """

    cbm_vars = cbm_variables.initialize_simulation_variables(
        classifiers, inventory, cbm.pool_codes, cbm.flux_indicator_codes
    )

    spinup_vars = cbm_variables.initialize_spinup_variables(
        cbm_vars, spinup_params, include_flux=spinup_reporting_func is not None
    )

    cbm.spinup(spinup_vars, reporting_func=spinup_reporting_func)
    cbm_vars = cbm.init(cbm_vars)
    reporting_func(0, cbm_vars)

    for time_step in range(1, int(n_steps) + 1):

        if pre_dynamics_func:
            cbm_vars = pre_dynamics_func(time_step, cbm_vars)

        cbm_vars = cbm.step(cbm_vars)
        reporting_func(time_step, cbm_vars)
