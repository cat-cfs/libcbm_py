from libcbm.model.cbm import cbm_variables


def simulate(cbm, n_steps, inventory, classifiers, pool_codes,
             flux_indicator_codes, pre_dynamics_func, reporting_func):
    """Runs the specified number of timesteps of the CBM model.  Model output
    is processed by the provided reporting_func. The provided
    pre_dynamics_func is called prior to each CBM dynamics step.

    Args:
        cbm (libcbm.model.cbm.cbm_model.CBM): Instance of the CBM model
        n_steps (int): The number of CBM timesteps to run
        inventory (pandas.DataFrame): CBM inventory which defines the initial
            state of the simulation
        classifiers (pandas.DataFrame): CBM classifiers for each of the rows
            in the inventory
        pool_codes (list): a list of strings describing each of the CBM pools
        flux_indicator_codes ([type]): [description]
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
    cbm.init(cbm_vars.inventory, cbm_vars.pools, cbm_vars.cbm_state)
    reporting_func(0, cbm_vars)
    for time_step in range(1, n_steps + 1):
        cbm_vars = pre_dynamics_func(cbm_vars)
        cbm.step(
            cbm_vars.inventory, cbm_vars.pools, cbm_vars.flux_indicators,
            cbm_vars.cbm_state, cbm_vars.cbm_params)
        reporting_func(time_step, cbm_vars)
