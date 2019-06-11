from libcbmwrapper import LibCBM_SpinupState
import numpy as np
import json, os,logging
class CBM3:
    def __init__(self, dll):
        self.dll = dll

        self.opNames = [
            "growth",
            "biomass_turnover",
            "snag_turnover",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]

        self.opProcesses = {
            "growth": 1,
            "biomass_turnover": 1,
            "snag_turnover": 1,
            "dom_decay": 2,
            "slow_decay": 2,
            "slow_mixing": 2,
            "disturbance": 3
        }


    def promoteScalar(self, value, size, dtype):
        '''
        if the specified value is scalar promote it to a numpy array filled with the scalar value
        otherwise return the value
        '''
        if value is None:
           return None
        if isinstance(value, np.ndarray):
            return value
        else:
            return np.ones(size, dtype=dtype) * value


    def spinup(self, pools, classifiers, inventory_age, spatial_unit,
               historic_disturbance_type, last_pass_disturbance_type,
               return_interval, min_rotations, max_rotations, delay,
               mean_annual_temp=None, debug_output_path = None):
        pools[:,0] = 1.0
        nstands = pools.shape[0]
        slow_pools_indices = [
            x["index"] for x in self.configuration["pools"] 
            if x["name"] in ["AboveGroundSlowSoil", "BelowGroundSlowSoil"]]
        #allocate working variables
        age = np.zeros(nstands, dtype=np.int32)
        slowPools = np.zeros(nstands, dtype=np.float)
        spinup_state = np.zeros(nstands, dtype=np.uint32)
        rotation = np.zeros(nstands, dtype=np.int32)
        step = np.zeros(nstands, dtype=np.int32)
        lastRotationSlowC = np.zeros(nstands, dtype=np.float)
        disturbance_types = np.zeros(nstands, dtype=np.int32)

        inventory_age = self.promoteScalar(inventory_age, nstands, dtype=np.int32)
        spatial_unit = self.promoteScalar(spatial_unit, nstands, dtype=np.int32)
        historic_disturbance_type = self.promoteScalar(historic_disturbance_type, nstands, dtype=np.int32)
        last_pass_disturbance_type = self.promoteScalar(last_pass_disturbance_type, nstands, dtype=np.int32)
        return_interval = self.promoteScalar(return_interval, nstands, dtype=np.int32)
        min_rotations = self.promoteScalar(min_rotations, nstands, dtype=np.int32)
        max_rotations = self.promoteScalar(max_rotations, nstands, dtype=np.int32)
        delay = self.promoteScalar(delay, nstands, dtype=np.int32)
        mean_annual_temp = self.promoteScalar(mean_annual_temp, nstands, dtype=np.float)
        logging.info("AllocateOp")
        ops = { x: self.dll.AllocateOp(nstands) for x in self.opNames }

        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["biomass_turnover"], ops["snag_turnover"],
                                spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"],
            ops["slow_mixing"], spatial_unit, mean_annual_temp)

        opSchedule = [
            "growth",
            "biomass_turnover",
            "snag_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]

        iteration = 0
        while (True):
            logging.info("AdvanceSpinupState")
            n_finished = self.dll.AdvanceSpinupState(
                return_interval, min_rotations, max_rotations, inventory_age,
                delay, slowPools, spinup_state, rotation, step,
                lastRotationSlowC)
            if n_finished == nstands:
                break
            logging.info("GetMerchVolumeGrowthOps")
            self.dll.GetMerchVolumeGrowthOps(ops["growth"],
                classifiers, pools, age, spatial_unit, None, None, None)

            #update state variables according to spinup state
            growing = ((spinup_state == LibCBM_SpinupState.HistoricalRotation) | 
                              (spinup_state == LibCBM_SpinupState.GrowToFinalAge))
            age[growing] = age[growing] + 1
            age[np.logical_not(growing)] = 0

            #set the historic/last pass disturbances
            historic_disturbance = (spinup_state == LibCBM_SpinupState.HistoricalDisturbance)
            disturbance_types[historic_disturbance] = historic_disturbance_type[historic_disturbance]

            last_disturbance = (spinup_state == LibCBM_SpinupState.LastPassDisturbance)
            disturbance_types[last_disturbance] = last_pass_disturbance_type[last_disturbance]
            
            #set the disturbance type for everything else as none
            disturbance_types[np.logical_not(np.logical_or(last_disturbance, historic_disturbance))] = -1

            logging.info("GetDisturbanceOps")
            self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit, disturbance_types)

            logging.info("ComputePools")
            self.dll.ComputePools([ops[x] for x in opSchedule], pools)

            #update the slow pools which are fed back into the spinup state
            slowPools = pools[:,slow_pools_indices[0]] + pools[:,slow_pools_indices[1]]

            iteration += 1


    def init(self, last_pass_disturbance_type, delay, inventory_age,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, land_class, age):

        self.dll.InitializeLandState(last_pass_disturbance_type, delay,
            inventory_age, last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, land_class, age)


    def step(self, step, pools, flux, classifiers, age, disturbance_types,
            spatial_unit, mean_annual_temp, transition_rule_ids,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, land_class,
            growth_multipliers, regeneration_delay):

        pools[:,0] = 1.0
        flux *= 0.0
        nstands = pools.shape[0]

        spatial_unit = self.promoteScalar(spatial_unit, nstands, dtype=np.int32)
        mean_annual_temp = self.promoteScalar(mean_annual_temp, nstands, dtype=np.int32)
        disturbance_types = self.promoteScalar(disturbance_types, nstands, dtype=np.int32)
        transition_rule_ids = self.promoteScalar(transition_rule_ids, nstands, dtype=np.int32)

        logging.info("AllocateOp")
        ops = { x: self.dll.AllocateOp(nstands) for x in self.opNames }

        opSchedule = [
            "disturbance",
            "growth",
            "biomass_turnover",
            "snag_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        logging.info("AdvanceStandState")
        self.dll.AdvanceStandState(classifiers, disturbance_types,
            transition_rule_ids, last_disturbance_type,
            time_since_last_disturbance, time_since_land_class_change,
            growth_enabled, land_class, regeneration_delay, age)

        growth_mult = np.where(growth_enabled == 0, 0.0, growth_multipliers)

        logging.info("GetDisturbanceOps")
        self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit,
                                  disturbance_types)

        logging.info("GetMerchVolumeGrowthOps")
        self.dll.GetMerchVolumeGrowthOps(ops["growth"],
            classifiers, pools, age, spatial_unit, last_disturbance_type,
            time_since_last_disturbance, growth_mult)
        
        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["biomass_turnover"], ops["snag_turnover"],
            spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"],
            ops["slow_mixing"], spatial_unit, mean_annual_temp)

        logging.info("Compute flux")
        self.dll.ComputeFlux([ops[x] for x in opSchedule], 
            [self.opProcesses[x] for x in opSchedule], pools, flux)

        age[regeneration_delay<=0]+=1
        regeneration_delay[regeneration_delay>0]-=1


        for x in self.opNames:
            self.dll.FreeOp(ops[x])
