import numpy as np
import pandas as pd
import json
import os


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

    def spinup(self, inventory, variables, parameters, debug=False):
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified carbon pools.  Each parameter has a first
        dimension of n_stands, with the exception of debug which is a bool.

        All ndarray parameters must have the ndarray.flags property set with
        the C_CONTIGUOUS attribute.

        Setting debug to true will return for each stand the time series of
        selected state variables used in the spinup procedure. has a
        significant negative impact on both CPU and memory performance.

        The only parameter modified by this function is "pools".

        All parameters other than pools and classifiers are subject to
        promotion, meaning if a scalar value is provided, that value will be
        repeated in a vector of length n_stands

        Arguments:
            pools {ndarray} -- a float64 matrix of shape (n_stands, n_pools)
                this paramater is assigned the result of CBM carbon pool
                spinup
            classifiers {ndarray} -- an int matrix of shape
                (n_stands, n_classifiers) Values are classifiers value ids
                referencing the classifier values stored in model parameters
                and configuration
            inventory_age {ndarray} or {int} -- int or int vector of length
                n_stands. Each value represents the age of a stand at the
                outset of CBM simulation.
            spatial_unit {ndarray} or {int} -- const, promotable int or int
                vector of length n_stands. Each value represent a key
                associated with several parameters stored in model parameters
                and configuration.
            afforestation_pre_type_id {ndarray} or {int} -- int or int vector
                of length n_stands. Each > zero value represents a key to a
                non-forest pre type, meaning the stand has pre-defined
                dead-organic-matter pools and will not be simulated by the
                spinup function. Zero or negative values represent a null
                pre-type and spinup will run.
            historic_disturbance_type {ndarray} or {int} -- int or int vector
                of length n_stands.  The disturbance type id used for each
                historical disturbance rotation in the spinup procedure.
            last_pass_disturbance_type {ndarray} or {int} -- int or int vector
                of length n_stands.  The disturbance type id used for the final
                disturbance in the spinup procedure.
            delay {ndarray} or {int} -- int or int vector of length n_stands.
                For stands deforested by the last pass disturbance type, this
                value indicates the number of years prior to the outset of
                CBM simulation that pass since the deforestation event in the
                spinup routine.

        Keyword Arguments:
            return_interval {ndarray} or {int} -- int or int vector
                of length n_stands. If specified, it defines the number of
                years in each historical disturbance rotation.
                (default: {None})
            min_rotations {ndarray} or {int} -- int or int vector
                of length n_stands. If specified, it defines the minimum number
                of historical rotations to perform. If unspecified a default
                value stored in the model parameters and configuration will be
                used. (default: {None})
            max_rotations {ndarray} or {int} -- int or int vector
                of length n_stands. If specified, it defines the maximum number
                of historical rotations to perform. If unspecified a default
                value stored in the model parameters and configuration will be
                used. (default: {None})
            mean_annual_temp  {ndarray} or {float64} -- float64 or float64
                vector of length n_stands.  If specified defines the mean
                annual temperature used in the spinup procedure.  If
                unspecified a default value stored in the model parameters and
                configuration will be used. (default: {None})
            debug {bool} -- if true this function will return a pandas
                dataframe of selected spinup state variables.
                (default: {False})
        Returns:
            pandas.DataFrame or None -- returns a debug dataframe if parameter
                debug is set to true, and None otherwise.
        """
        variables.pools[:, 0] = 1.0
        n_stands = variables.pools.shape[0]

        ops = {x: self.dll.AllocateOp(n_stands) for x in self.opNames}

        self.dll.GetTurnoverOps(ops["snag_turnover"], ops["biomass_turnover"],
                                inventory.spatial_unit)

        self.dll.GetDecayOps(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory.spatial_unit, True, parameters.mean_annual_temp)

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
                inventory.spatial_unit, parameters.return_interval,
                parameters.min_rotations, parameters.max_rotations,
                inventory.age, inventory.delay, variables.slowPools,
                inventory.historic_disturbance_type,
                inventory.last_pass_disturbance_type,
                inventory.afforestation_pre_type_id, variables.spinup_state,
                variables.disturbance_types, variables.rotation,
                variables.step, variables.lastRotationSlowC,
                variables.enabled)
            if n_finished == n_stands:
                break

            self.dll.GetMerchVolumeGrowthOps(
                ops["growth"], inventory.classifiers.to_numpy(),
                variables.pools, variables.age, inventory.spatial_unit,
                None, None, None, variables.growth_enabled)

            self.dll.GetDisturbanceOps(
                ops["disturbance"], inventory.spatial_unit,
                variables.disturbance_types)

            self.dll.ComputePools(
                [ops[x] for x in opSchedule], variables.pools,
                variables.enabled)

            self.dll.EndSpinupStep(
                variables.spinup_state, variables.pools,
                variables.disturbance_types, variables.age,
                variables.slowPools, variables.growth_enabled)

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
                    "disturbance_type": variables.disturbance_types
                    }))

            iteration = iteration + 1

        for x in self.opNames:
            self.dll.FreeOp(ops[x])
        return debug_output

    def init(self, inventory, variables):
        """Set the initial state of CBM variables after spinup and prior to
        starting CBM simulation.

        Several variables reference the "model parameters and configuration"
        which are passed to the LibCBMWrapper initialization methods.

        All ndarray parameters must have the ndarray.flags property set with
        the C_CONTIGUOUS attribute.

        In the following documentation
         - "const" indicates a parameter will not be modified by this function
         - "promotable" indicates if a scalar value is passed it will be
           repeated in a vector of length n_stands

        Arguments:
            last_pass_disturbance_type {ndarray} -- const, promotable int, or
                int vector of length n_stands. Defines the most recent
                disturbance that occurred in CBM. In the case that this
                disturbance is a deforestation type, the initial landclass
                will be set by this method.
            delay {ndarray} -- const, promotable int, or int vector of length
                n_stands.  The number of timesteps that have elapsed since a
                deforestation event occurred, since certain land classes expire
                after a timestep limit, this is used to walk through the
                transitional landclasses and arrive at the initial UNFCCC land
                class state.
            inventory_age {ndarray} -- const, promotable int, or int vector of
                length n_stands.  The number of timesteps that have elapsed
                since a non-deforestation last pass disturbance type has
                occurred.  Used to assign the initial inventory age.
            spatial_unit {ndarray} or {int} -- const, promotable int or int
                vector of length n_stands. Each value represent a key
                associated with several parameters stored in model parameters
                and configuration.
            afforestation_pre_type_id {ndarray} -- const, promotable int, or
                int vector of length n_stands.  When set to a valid
                afforestation pre-type, the last pass disturbance type is
                ignored, and the initial pool values are set by this method.
                The enabled flag is also set to 0, meaning that at the start
                of CBM simulation, the stand will not be simulated, until an
                afforestation event occurs.
            pools {ndarray} -- a float64 matrix of shape (n_stands, n_pools)
                this paramater is assigned by this method when
                afforestation_pre_type_id is > 0
            last_disturbance_type {ndarray} -- int vector of length
                n_stands. Set to the last_pass_disturbance_type.
            time_since_last_disturbance {ndarray} -- int vector of length
                n_stands. Set based on the inventory_age or delay values to
                the number of timesteps since a disturbance last occurred.
            time_since_land_class_change {ndarray} -- int vector of length
                n_stands. Set to the number of timesteps since a land-class
                changing disturbance event (if any) occurred, and otherwise 0.
            growth_enabled {ndarray} -- int vector of length n_stands. Assigned
                based on the value of last_disturbance_type. For example in the
                case of deforestation, this will be set to 0.
            enabled {ndarray} -- int vector of length n_stands. Assigned for
                cases where no CBM simulation should occur. For example for
                peatlands or pre-afforestation land this will be set to 0.
            land_class {ndarray} -- int vector of length n_stands. Set to the
                deforestation related land class id, if a deforestation
                disturbance type id is used, and otherwise not modified by
                this function.
            age {ndarray} -- int vector of length n_stands. Set to the
                inventory_age value, unless last_disturbance_type is a
                deforestation type, or afforestation_pre_type_id is a valid
                afforestation pre-type-id.  In these deforestation or
                afforestation cases a non-zero inventory_age also triggers an
                error.
        """

        self.dll.InitializeLandState(
            inventory.last_pass_disturbance_type, inventory.delay,
            inventory.age, inventory.spatial_unit,
            inventory.afforestation_pre_type_id, variables.pools,
            variables.state.last_disturbance_type,
            variables.state.time_since_last_disturbance,
            variables.state.time_since_land_class_change,
            variables.state.growth_enabled, variables.state.enabled,
            variables.state.land_class, variables.state.age)

    def step(self, inventory, variables, parameters):
        """Advances the specified CBM variables through one time step of CBM
        simulation.


        Arguments:
            inventory {pandas.DataFrame} --
            variables {[type]} -- [description]
            parameters {[type]} -- [description]
        """

        variables.pools[:, 0] = 1.0
        variables.flux *= 0.0
        n_stands = variables.pools.shape[0]

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

        self.dll.AdvanceStandState(
            inventory.classifiers.to_numpy(), inventory.spatial_unit,
            parameters.disturbance_type, parameters.transition_rule_id,
            variables.state.last_disturbance_type,
            variables.state.time_since_last_disturbance,
            variables.state.time_since_land_class_change,
            variables.state.growth_enabled, variables.state.enabled,
            variables.state.land_class, variables.state.regeneration_delay,
            variables.state.age)

        self.dll.GetDisturbanceOps(
            ops["disturbance"], inventory.spatial_unit,
            parameters.disturbance_type)

        self.dll.ComputeFlux(
            [ops["disturbance"]], [self.opProcesses["disturbance"]],
            variables.pools, variables.flux, enabled=None)

        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)

        self.dll.GetMerchVolumeGrowthOps(
            ops["growth"], inventory.classifiers.to_numpy(), inventory.pools,
            variables.state.age, inventory.spatial_unit,
            variables.state.last_disturbance_type,
            variables.state.time_since_last_disturbance,
            variables.state.growth_multiplier, variables.state.growth_enabled)

        self.dll.GetTurnoverOps(
            ops["snag_turnover"], ops["biomass_turnover"],
            inventory.spatial_unit)

        self.dll.GetDecayOps(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            inventory.spatial_unit, parameters.mean_annual_temp)

        self.dll.ComputeFlux(
            [ops[x] for x in annual_process_opSchedule],
            [self.opProcesses[x] for x in annual_process_opSchedule],
            variables.pools, variables.flux, variables.state.enabled)

        self.dll.EndStep(
            variables.state.age, variables.state.regeneration_delay,
            variables.state.growth_enabled)

        for x in self.opNames:
            self.dll.FreeOp(ops[x])
