import numpy as np
import pandas as pd
import json
import os
from types import SimpleNamespace


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

        self.opNames = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]

        self.opProcesses = {
            "growth": 1,
            "snag_turnover": 1,
            "biomass_turnover": 1,
            "dom_decay": 2,
            "slow_decay": 2,
            "slow_mixing": 2,
            "disturbance": 3
        }

    def spinup(self, inventory, pools, variables, parameters, debug=False):
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified variables.

        Args:
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
            x: self.compute_functions.AllocateOp(n_stands)
            for x in self.opNames}

        self.model_functions.GetTurnoverOps(
            ops["snag_turnover"], ops["biomass_turnover"], inventory)

        self.model_functions.GetDecayOps(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory, parameters, historical_mean_annual_temp=True)

        opSchedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]
        debug_output = None
        if(debug):
            debug_output = pd.DataFrame()
        iteration = 0

        while (True):

            n_finished = self.model_functions.AdvanceSpinupState(
                inventory, variables, parameters)

            if n_finished == n_stands:
                break

            self.model_functions.GetMerchVolumeGrowthOps(
                ops["growth"], inventory, pools, variables)

            self.model_functions.GetDisturbanceOps(
                ops["disturbance"], inventory, variables)

            self.compute_functions.ComputePools(
                [ops[x] for x in opSchedule], pools,
                variables.enabled)

            self.model_functions.EndSpinupStep(pools, variables)

            if(debug):
                debug_output = debug_output.append(pd.DataFrame(data={
                    "index": list(range(n_stands)),
                    "iteration": iteration,
                    "age": variables.age,
                    "slow_pools": variables.slowPools,
                    "spinup_state": variables.spinup_state,
                    "rotation": variables.rotation,
                    "last_rotation_c": variables.lastRotationSlowC,
                    "step": variables.step,
                    "disturbance_type": variables.disturbance_type
                    }))

            iteration = iteration + 1

        for x in self.opNames:
            self.compute_functions.FreeOp(ops[x])
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

        self.model_functions.InitializeLandState(
            inventory, pools, state_variables)

    def step(self, inventory, pools, flux, state_variables, parameters):
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
            x: self.compute_functions.AllocateOp(n_stands)
            for x in self.opNames}

        annual_process_opSchedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        self.model_functions.AdvanceStandState(
            inventory, state_variables, parameters)

        self.model_functions.GetDisturbanceOps(
            ops["disturbance"], inventory, parameters)

        self.compute_functions.ComputeFlux(
            [ops["disturbance"]], [self.opProcesses["disturbance"]],
            pools, flux, enabled=None)

        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)

        self.model_functions.GetMerchVolumeGrowthOps(
            ops["growth"], inventory, pools, state_variables)

        self.model_functions.GetTurnoverOps(
            ops["snag_turnover"], ops["biomass_turnover"],
            inventory)

        self.model_functions.GetDecayOps(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory, parameters)

        self.compute_functions.ComputeFlux(
            [ops[x] for x in annual_process_opSchedule],
            [self.opProcesses[x] for x in annual_process_opSchedule],
            pools, flux, state_variables.enabled)

        self.model_functions.EndStep(state_variables)

        for x in self.opNames:
            self.compute_functions.FreeOp(ops[x])
