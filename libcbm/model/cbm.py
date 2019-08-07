import numpy as np
import pandas as pd
import json
import os
from types import SimpleNamespace


class CBM:
    def __init__(self, dll, config):
        """Creates a new instance of the CBM model with the specified
        LibCBM wrapper instance. The wrapper instance is initialized
        with model parameters and configuration.

        Arguments:
            dll {LibCBMWrapper} -- an instance of the LibCBMWrapper.
        """
        self.dll = dll
        self.config = config

        config_string = json.dumps(config)
        dll.InitializeCBM(config_string)

        # create an index for lookup of classifiers
        classifier_id_lookup = {x["id"]: x for x in config["classifiers"]}
        self.classifier_lookup = {}
        for cv in config["classifier_values"]:
            classifier_id = cv["classifier_id"]
            classifier_name = classifier_id_lookup[classifier_id]["name"]
            if classifier_name in self.classifier_lookup:
                self.classifier_lookup[classifier_name][cv["value"]] = cv
            else:
                self.classifier_lookup[classifier_name] = {cv["value"]: cv}

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

    def get_classifier_value_id(self, classifier_name, classifier_value_name):
        """Get the classifier value id associated with the classifier_name,
        classifier_value_name pair

        Arguments:
            classifier {str} -- name of the classifier
            classifier_value {str} -- name of the classifier value

        Returns:
            int -- identifier for the classifier/classifier value
        """
        c = self.classifier_lookup[classifier_name]
        cv = c[classifier_value_name]
        return cv["id"]

    def spinup(self, inventory, pools, variables, parameters, debug=False):
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified variables.

        See libcbm.model.cbm_variables for creating this function's
        parameters.

        Arguments:
            inventory {object} -- Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
            pools {object} -- CBM pools (of dimension n_stands by n_pools)
                initialized with spinup carbon values by this function
            variables {object} -- spinup working variables
            parameters {object} -- spinup parameters

        Keyword Arguments:
            debug {bool} -- if true this function will return a pandas
                dataframe of selected spinup state variables.
                (default: {False})

        Returns:
            pandas.DataFrame or None -- returns a debug dataframe if parameter
                debug is set to true, and None otherwise.
        """

        n_stands = pools.shape[0]

        ops = {x: self.dll.AllocateOp(n_stands) for x in self.opNames}

        self.dll.GetTurnoverOps(ops["snag_turnover"], ops["biomass_turnover"],
                                inventory.spatial_unit)

        self.dll.GetDecayOps(
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

            n_finished = self.dll.AdvanceSpinupState(
                inventory, variables, parameters)

            if n_finished == n_stands:
                break

            self.dll.GetMerchVolumeGrowthOps(
                ops["growth"], inventory, variables.pools, variables)

            self.dll.GetDisturbanceOps(
                ops["disturbance"], inventory, variables)

            self.dll.ComputePools(
                [ops[x] for x in opSchedule], pools,
                variables.enabled)

            self.dll.EndSpinupStep(pools, variables)

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
            self.dll.FreeOp(ops[x])
        return debug_output

    def init(self, inventory, pools, state_variables):
        """Set the initial state of CBM variables after spinup and prior
        to starting CBM simulation

        See libcbm.model.cbm_variables for creating this function's
        parameters.

        Arguments:
            inventory {object} -- Read-only data comprised of classifier sets
                and cbm inventory data
            pools {object} -- CBM pools (of dimension n_stands by n_pools)
                initialized with non-forest initial carbon values by this
                function where applicable
            state_variables {object} -- state variables to be initialized by
                this function for the start of CBM simulation

        """
        self.dll.InitializeLandState(
            inventory, pools, state_variables)

    def step(self, inventory, pools, flux, state_variables, parameters):
        """Advances the specified CBM variables through one time step of CBM
        simulation.

        See libcbm.model.cbm_variables for creating this function's
        parameters.

        Arguments:
            inventory {pandas.DataFrame} -- Data comprised of classifier sets
                and cbm inventory data.  Inventory data will not be modified
                by this function, but classifier sets may be modified if
                transition rules are used.
            pools {pandas.DataFrame or numpy.ndarray} -- ...
            flux {pandas.DataFrame or numpy.ndarray} -- ...
            state_variables {pandas.DataFrame} -- simulation variables which
                define all non-pool state in the CBM model.  Altered by this
                function call.
            parameters {object} -- read-only parameters used in a CBM timestep:
                - disturbance types
                - mean annual temperature
                - transitions
        """

        flux *= 0.0
        n_stands = pools.shape[0]

        ops = {x: self.dll.AllocateOp(n_stands) for x in self.opNames}

        annual_process_opSchedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        self.dll.AdvanceStandState(inventory, state_variables, parameters)

        self.dll.GetDisturbanceOps(
            ops["disturbance"], inventory, parameters)

        self.dll.ComputeFlux(
            [ops["disturbance"]], [self.opProcesses["disturbance"]],
            pools, flux, enabled=None)

        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)

        self.dll.GetMerchVolumeGrowthOps(
            ops["growth"], inventory, pools, state_variables)

        self.dll.GetTurnoverOps(
            ops["snag_turnover"], ops["biomass_turnover"],
            inventory)

        self.dll.GetDecayOps(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory, parameters)

        self.dll.ComputeFlux(
            [ops[x] for x in annual_process_opSchedule],
            [self.opProcesses[x] for x in annual_process_opSchedule],
            pools, flux, state_variables.enabled)

        self.dll.EndStep(state_variables)

        for x in self.opNames:
            self.dll.FreeOp(ops[x])
