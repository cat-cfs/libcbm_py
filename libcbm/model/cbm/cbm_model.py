import pandas as pd
from libcbm import data_helpers


def get_op_names():
    """Gets the names of the CBM dynamics operations

    Returns:
        list: the list of string names describing the dynamics ops
    """
    return [
        "growth",
        "snag_turnover",
        "biomass_turnover",
        "overmature_decline",
        "dom_decay",
        "slow_decay",
        "slow_mixing",
        "disturbance"]


def get_op_processes():
    """Gets a dictionary of operation name to process id, which is
    used to group flux indicators

    Returns:
        dict: dictionary of key: operation name, value: process id
    """
    return {
        "growth": 1,
        "overmature_decline": 1,
        "snag_turnover": 1,
        "biomass_turnover": 1,
        "dom_decay": 2,
        "slow_decay": 2,
        "slow_mixing": 2,
        "disturbance": 3}


class CBM:
    """The CBM model.

    Args:
        compute_functions (libcbm.wrapper.libcbm_wrapper.LibCBMWrapper): an
            instance of LibCBMWrapper.
        model_functions (libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper): an
            instance of CBMWrapper
    """
    def __init__(self, compute_functions, model_functions):

        self.compute_functions = compute_functions
        self.model_functions = model_functions

        self.op_names = get_op_names()

        self.op_processes = get_op_processes()

    def spinup(self, classifiers, inventory, pools, variables, parameters,
              debug=False):
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified variables.

        Args:
            classifiers (pandas.DataFrame):
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition
            pools (pandas.DataFrame or numpy.ndarray): CBM pools of
                dimension n_stands by n_pools. Initialized with spinup carbon
                values by this function.  Column order is important. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_pools` for
                a compatible definition
            variables (object): Spinup working variables.  Defines all
                non-pool simulation state during spinup.  See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_variables`
                for a compatible definition
            parameters (object): spinup parameters. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_parameters`
                for a compatible definition
            debug (bool, optional) If true this function will return a pandas
                dataframe of selected spinup state variables. Defaults to
                False.

        Returns:
            pandas.DataFrame or None: returns a debug dataframe if parameter
                debug is set to true, and None otherwise.
        """

        n_stands = pools.shape[0]

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names}

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"], inventory)

        self.model_functions.get_decay_ops(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory, parameters, historical_mean_annual_temp=True)

        op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]
        debug_output = None
        iteration = 0

        while True:

            n_finished = self.model_functions.advance_spinup_state(
                inventory, variables, parameters)

            if n_finished == n_stands:
                break

            self.model_functions.get_merch_volume_growth_ops(
                ops["growth"], ops["overmature_decline"], classifiers,
                inventory, pools, variables)

            self.model_functions.get_disturbance_ops(
                ops["disturbance"], inventory, variables)

            self.compute_functions.compute_pools(
                [ops[x] for x in op_schedule], pools,
                variables.enabled)

            self.model_functions.end_spinup_step(pools, variables)

            if debug:
                debug_output = data_helpers.append_simulation_result(
                    debug_output,
                    pd.DataFrame(
                        data={
                            "age": variables.age,
                            "slow_pools": variables.slowPools,
                            "spinup_state": variables.spinup_state,
                            "rotation": variables.rotation,
                            "last_rotation_c": variables.lastRotationSlowC,
                            "step": variables.step,
                            "disturbance_type": variables.disturbance_type
                        }),
                    iteration)

            iteration = iteration + 1

        for x in self.op_names:
            self.compute_functions.free_op(ops[x])
        return debug_output

    def init(self, inventory, pools, state_variables):
        """Set the initial state of CBM variables after spinup and prior
        to starting CBM simulation stepping

        Args:
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition
            pools (pandas.DataFrame or numpy.ndarray): -- CBM pools of
                dimension n_stands by n_pools. Pool values are set by this
                function for the case of pre-afforestation soil types.
                Column order is important. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_pools` for
                a compatible definition
            state_variables (pandas.DataFrame): -- Simulation variables which
                define all non-pool state in the CBM model.  Altered by this
                function call.  See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition
        """

        # the following line is tricky, and needs some more thought:
        # 1. defining the landclass as FL_FL is not problematic for regular
        #    forest cases.
        # 2. defining the landclass as a non forest landclass is not
        #    problematic for afforestation, and using a forested landclass
        #    is problematic.  This is handled by libcbm, in that an error is
        #    thrown if a forested land class is used with an afforestation
        #    pre-type.
        # 3. using a deforestation disturbance for
        #    inventory.last_pass_disturbance_type type should overwrite the
        #    state variable landclass since each deforestation disturbance
        #    type is explicitly associated with a transitional land class.
        #    This means that inventory.last_pass_disturbance_type along with
        #    inventory.delay indicates the correct value for
        #    state_variables.land_class at CBM startup, and it can potentially
        #    contradict the value of inventory.land_class. libcbm does not
        #    attempt to throw an error if this situation is detected.
        state_variables.land_class = inventory.land_class

        self.model_functions.initialize_land_state(
            inventory, pools, state_variables)

    def step(self, classifiers, inventory, pools, flux, state_variables,
             parameters):
        """Advances the specified CBM variables through one time step of CBM
        simulation.

        Args:
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition
            pools (pandas.DataFrame or numpy.ndarray): CBM pools of
                dimension n_stands by n_pools. Set with the result of pool
                Carbon dynamics for this timestep.  Column order is important.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_pools` for
                a compatible definition
            flux (pandas.DataFrame or numpy.ndarray): CBM flux values of
                dimension n_stands by n_flux_indicators. Set with the flux
                indicator values for pool flows that occur in this timestep.
                Column order is important. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_flux` for a
                compatible definition.
            state_variables (pandas.DataFrame): simulation variables which
                define all non-pool state in the CBM model.  Altered by this
                function call.  See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition
            parameters (object): Read-only parameters used in a CBM timestep.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_parameters`
                for a compatible definition.
        """

        # zero the memory (simply using flux *= 0.0 caused a copy
        # with a change in contiguity in some cases!)
        if isinstance(flux, pd.DataFrame):
            flux.values[:] = 0
        else:
            flux[:] = 0

        n_stands = pools.shape[0]

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names}

        annual_process_op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        self.model_functions.advance_stand_state(
            classifiers, inventory, state_variables, parameters)

        self.model_functions.get_disturbance_ops(
            ops["disturbance"], inventory, parameters)

        self.compute_functions.compute_flux(
            [ops["disturbance"]], [self.op_processes["disturbance"]],
            pools, flux, enabled=None)

        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)

        self.model_functions.get_merch_volume_growth_ops(
            ops["growth"], ops["overmature_decline"], classifiers, inventory,
            pools, state_variables)

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"],
            inventory)

        self.model_functions.get_decay_ops(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory, parameters)

        self.compute_functions.compute_flux(
            [ops[x] for x in annual_process_op_schedule],
            [self.op_processes[x] for x in annual_process_op_schedule],
            pools, flux, state_variables.enabled)

        self.model_functions.end_step(state_variables)

        for x in self.op_names:
            self.compute_functions.free_op(ops[x])
