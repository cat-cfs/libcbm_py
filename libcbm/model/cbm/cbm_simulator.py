from types import SimpleNamespace
from libcbm import data_helpers
from libcbm.model.cbm import cbm_variables


def create_in_memory_reporting_func():
    """Create storage and a function for simulation results.  The function
    return value can be passed to :py:func:`simulate` to track simultion
    results.

    Returns:
        tuple: a pair of values:

            1. types.SimpleNameSpace: an object with properties:

              - pool_indicators a pandas.DataFrame for storing pools
              - flux_indicators a pandas.DataFrame for storing fluxes
              - state_indicators a pandas.DataFrame for storing state

            2. func: a function for appending to the above results dataframes

    """
    results = SimpleNamespace()
    results.pool_indicators = None
    results.flux_indicators = None
    results.state_indicators = None

    def append_simulation_result(timestep, cbm_vars):
        results.pool_indicators = data_helpers.append_simulation_result(
            results.pool_indicators, cbm_vars.pools, timestep)
        if timestep > 0:
            results.flux_indicators = data_helpers.append_simulation_result(
                results.flux_indicators, cbm_vars.flux_indicators, timestep)
        results.state_indicators = data_helpers.append_simulation_result(
            results.state_indicators, cbm_vars.state, timestep)

    return results, append_simulation_result


def simulate(cbm, n_steps, classifiers, inventory, pool_codes,
             flux_indicator_codes, pre_dynamics_func, reporting_func):
    """Runs the specified number of timesteps of the CBM model.  Model output
    is processed by the provided reporting_func. The provided
    pre_dynamics_func is called prior to each CBM dynamics step.

    Args:
        cbm (libcbm.model.cbm.cbm_model.CBM): Instance of the CBM model
        n_steps (int): The number of CBM timesteps to run
        classifiers (pandas.DataFrame): CBM classifiers for each of the rows
            in the inventory
        inventory (pandas.DataFrame): CBM inventory which defines the initial
            state of the simulation
        pool_codes (list): a list of strings describing each of the CBM pools
        flux_indicator_codes (list): a list of strings describing the CBM flux
            indicators.
        pre_dynamics_func (function): A function which both accepts and
            returns all CBM variables.  The layout of the CBM variables is the
            same as the return value of
            :py:func:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        reporting_func (function): a function which accepts all CBM variables.
            The layout of the CBM variables is the same as the return value of
            :py:func:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
    """
    n_stands = inventory.shape[0]

    spinup_params = cbm_variables.initialize_spinup_parameters(n_stands)
    spinup_variables = cbm_variables.initialize_spinup_variables(n_stands)

    cbm_vars = cbm_variables.initialize_simulation_variables(
        classifiers, inventory, pool_codes, flux_indicator_codes)

    cbm.spinup(
        cbm_vars.inventory, cbm_vars.pools, spinup_variables, spinup_params)
    cbm.init(cbm_vars.inventory, cbm_vars.pools, cbm_vars.state)
    reporting_func(0, cbm_vars)
    for time_step in range(1, n_steps + 1):
        cbm_vars = pre_dynamics_func(cbm_vars)
        cbm.step(
            cbm_vars.inventory, cbm_vars.pools, cbm_vars.flux_indicators,
            cbm_vars.state, cbm_vars.params)
        reporting_func(time_step, cbm_vars)
