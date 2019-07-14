import numpy as np
import pandas as pd
import json, os,logging

class CBM:
    def __init__(self, dll):
        """Creates a new instance of the CBM model with the specified LibCBM wrapper instance.
        The wrapper instance is initialized with model parameters and configuration.
        
        Arguments:
            dll {LibCBMWrapper} -- an instance of the LibCBMWrapper.
        """
        self.dll = dll

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


    def promote_scalar(self, value, size, dtype):
        """If the specified value is scalar promote it to a numpy array filled 
        with the scalar value, and otherwise return the value.  This is purely 
        a helper function to allow scalar parameters for certain vector
        functions
        
        Arguments:
            value {ndarray} or {number} or {None} -- 
            size {int} -- the length of the resulting vector if promotion
            occurs
            dtype {object} -- object used to define the type of the resulting
            vector if promotion occurs
        
        Returns:
            ndarray or None -- returns either the original value, a promoted 
            scalar or None depending on the specified values
        """
        if value is None:
           return None
        if isinstance(value, np.ndarray):
            return value
        else:
            return np.ones(size, dtype=dtype) * value


    def spinup(self, pools, classifiers, inventory_age,
               spatial_unit, afforestation_pre_type_id,
               historic_disturbance_type, last_pass_disturbance_type,
               delay, return_interval=None, min_rotations=None,
               max_rotations=None, mean_annual_temp=None, debug=False):
        """Run the CBM-CFS3 spinup function on an array of stands, 
        initializing the specified carbon pools.  Each parameter has a first
        dimension of n_stands, with the exception of debug which is a bool.
        
        setting debug to true will return for each stand the time series of
        selected state variables used in the spinup procedure. has a 
        significant negative impact on both CPU and memory performance.

        The only parameter modified by this function is "pools".  

        All parameters other than pools and classifiers are subject to
        promotion, meaning if a scalar value is provided, that value will be
        repeated in a vector of length n_stands
        
        Arguments:
            pools {ndarray} -- a float64 matrix of shape (n_stands, n_pools) this
                paramater is assigned the result of CBM carbon pool spinup
            classifiers {ndarray} -- an int matrix of shape 
                (n_stands, n_classifiers) Values are classifiers value ids
                referencing the classifier values stored in model parameters
                and configuration
            inventory_age {ndarray} or {int} -- int or int vector of length
                n_stands. Each value represents the age of a stand at the
                outset of CBM simulation.
            spatial_unit {ndarray} or {int} -- int or int vector of length
                n_stands. Each value represent a key several parameters stored
                in model parameters and configuration
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
        pools[:,0] = 1.0
        n_stands = pools.shape[0]

        #allocate working variables
        age = np.zeros(n_stands, dtype=np.int32)
        slowPools = np.zeros(n_stands, dtype=np.float)
        spinup_state = np.zeros(n_stands, dtype=np.uint32)
        rotation = np.zeros(n_stands, dtype=np.int32)
        step = np.zeros(n_stands, dtype=np.int32)
        lastRotationSlowC = np.zeros(n_stands, dtype=np.float)
        disturbance_types = np.zeros(n_stands, dtype=np.int32)
        growth_enabled = np.ones(n_stands, dtype=np.int32)
        enabled = np.ones(n_stands, dtype=np.int32)

        inventory_age = self.promote_scalar(inventory_age, n_stands, dtype=np.int32)
        spatial_unit = self.promote_scalar(spatial_unit, n_stands, dtype=np.int32)
        historic_disturbance_type = self.promote_scalar(historic_disturbance_type, n_stands, dtype=np.int32)
        last_pass_disturbance_type = self.promote_scalar(last_pass_disturbance_type, n_stands, dtype=np.int32)
        return_interval = self.promote_scalar(return_interval, n_stands, dtype=np.int32)
        min_rotations = self.promote_scalar(min_rotations, n_stands, dtype=np.int32)
        max_rotations = self.promote_scalar(max_rotations, n_stands, dtype=np.int32)
        delay = self.promote_scalar(delay, n_stands, dtype=np.int32)
        mean_annual_temp = self.promote_scalar(mean_annual_temp, n_stands, dtype=np.float)

        logging.info("AllocateOp")
        ops = { x: self.dll.AllocateOp(n_stands) for x in self.opNames }

        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["snag_turnover"], ops["biomass_turnover"],
                                spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"],
            ops["slow_mixing"], spatial_unit, True, mean_annual_temp)

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
            logging.info("AdvanceSpinupState")

            n_finished = self.dll.AdvanceSpinupState(
                spatial_unit, return_interval, min_rotations, max_rotations,
                inventory_age, delay, slowPools, historic_disturbance_type,
                last_pass_disturbance_type, afforestation_pre_type_id,
                spinup_state, disturbance_types, rotation, step,
                lastRotationSlowC, enabled)
            if n_finished == n_stands:
                break
            logging.info("GetMerchVolumeGrowthOps")
            self.dll.GetMerchVolumeGrowthOps(ops["growth"],
                classifiers, pools, age, spatial_unit, None, None, None,
                growth_enabled)

            logging.info("GetDisturbanceOps")
            self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit,
                                       disturbance_types)

            logging.info("ComputePools")
            self.dll.ComputePools([ops[x] for x in opSchedule], pools, enabled)

            self.dll.EndSpinupStep(spinup_state, pools, disturbance_types,
                age, slowPools, growth_enabled)
            if(debug):
                debug_output = debug_output.append(pd.DataFrame(data={
                    "index": list(range(n_stands)),
                    "iteration": iteration,
                    "age": age,
                    "slow_pools": slowPools,
                    "spinup_state": spinup_state,
                    "rotation": rotation,
                    "last_rotation_c": lastRotationSlowC,
                    "step": step,
                    "disturbance_type": disturbance_types
                    }))

            iteration = iteration + 1
        for x in self.opNames:
            self.dll.FreeOp(ops[x])
        return debug_output

    def init(self, last_pass_disturbance_type, delay, inventory_age,
             spatial_unit, afforestation_pre_type_id, pools, last_disturbance_type,
             time_since_last_disturbance, time_since_land_class_change,
             growth_enabled, enabled, land_class, age):

        self.dll.InitializeLandState(last_pass_disturbance_type, delay,
            inventory_age, spatial_unit, afforestation_pre_type_id, pools,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled, land_class,
            age)


    def step(self, pools, flux, classifiers, age, disturbance_type,
            spatial_unit, mean_annual_temp, transition_rule_id,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled, land_class,
            growth_multiplier, regeneration_delay):
        """Advances the specified arguments through one time step of CBM simulation.
        
        In the following documentation
         - "const" indicates a parameter will not be modified by this function
         - "promotable" indicates if a scalar value is passed it will be repeated in a vector of length n_stands

        Arguments:
            pools {ndarray} -- a float64 matrix of shape (n_stands, n_pools) 
                Assigned the result of the CBM timestep carbon dynamics
            flux {ndarray} -- a float64 matrix of shape 
                (n_stands, n_flux_indicators) Assigned the flux indicators, as
                configured in the model parameters and configuration which 
                occur in the timestep carbon dynamics
            classifiers {ndarray} -- an int matrix of shape 
                (n_stands, n_classifiers) Values are classifiers value ids
                referencing the classifier values in the model parameters and
                configuration.
            age {ndarray} -- int vector of length n_stands.  The model will 
                update age depending on stand state and disturbances
            disturbance_type {ndarray} -- const, promotable int, or int vector of
                length n_stands. The vector of disturbance type ids to apply 
                to stands in this step. Negative or 0 values indicate no 
                disturbance, and > 0 values reference the disturbance type
                parameters stored in the model parameters and configuration.
            spatial_unit {ndarray} or {int} -- const, promotable int or int 
                vector of length n_stands. Each value represent a key several
                parameters stored in model parameters and configuration.
            mean_annual_temp {ndarray} or {float64} -- const, promotable float64
                or float64 vector of length n_stands. Each value is used in the
                computation of dead organic matter decay rates.
            transition_rule_ids {[type]} -- [description]
            last_disturbance_type {[type]} -- [description]
            time_since_last_disturbance {[type]} -- [description]
            time_since_land_class_change {[type]} -- [description]
            growth_enabled {[type]} -- [description]
            enabled {[type]} -- [description]
            land_class {[type]} -- [description]
            growth_multiplier {[type]} -- [description]
            regeneration_delay {[type]} -- [description]
        """
        

        pools[:,0] = 1.0
        flux *= 0.0
        nstands = pools.shape[0]

        spatial_unit = self.promote_scalar(spatial_unit, nstands, dtype=np.int32)
        mean_annual_temp = self.promote_scalar(mean_annual_temp, nstands, dtype=np.int32)
        disturbance_type = self.promote_scalar(disturbance_type, nstands, dtype=np.int32)
        transition_rule_id = self.promote_scalar(transition_rule_id, nstands, dtype=np.int32)

        logging.info("AllocateOp")
        ops = { x: self.dll.AllocateOp(nstands) for x in self.opNames }

        annual_process_opSchedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        logging.info("AdvanceStandState")
        self.dll.AdvanceStandState(classifiers, spatial_unit,
            disturbance_type, transition_rule_id, last_disturbance_type,
            time_since_last_disturbance, time_since_land_class_change,
            growth_enabled, enabled, land_class, regeneration_delay, age)

        logging.info("GetDisturbanceOps")
        self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit,
                                  disturbance_type)
        logging.info("Compute flux")
        self.dll.ComputeFlux([ops["disturbance"]],
            [self.opProcesses["disturbance"]], pools, flux,
            enabled=None)
        #enabled = none on line above is due to a possible bug in CBM3. This is very much an edge case:
        #stands can be disturbed despite having other C-dynamics processes disabled
        #(which happens in peatland, cropland, and other non forest landclasses)

        logging.info("GetMerchVolumeGrowthOps")
        self.dll.GetMerchVolumeGrowthOps(ops["growth"],
            classifiers, pools, age, spatial_unit, last_disturbance_type,
            time_since_last_disturbance, growth_multiplier, growth_enabled)
        
        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["snag_turnover"], ops["biomass_turnover"],
            spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"],
            ops["slow_mixing"], spatial_unit, mean_annual_temp)

        logging.info("Compute flux")
        self.dll.ComputeFlux([ops[x] for x in annual_process_opSchedule],
            [self.opProcesses[x] for x in annual_process_opSchedule],
            pools, flux, enabled)

        self.dll.EndStep(age, regeneration_delay, growth_enabled)
        for x in self.opNames:
            self.dll.FreeOp(ops[x])
