import numpy as np
import pandas as pd
import json, os,logging

class CBM:
    def __init__(self, dll):
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


    def spinup(self, pools, classifiers, inventory_age,
               spatial_unit, afforestation_pre_type_id,
               historic_disturbance_type, last_pass_disturbance_type,
               delay, return_interval=None, min_rotations=None,
               max_rotations=None, mean_annual_temp=None, debug=False):
        pools[:,0] = 1.0
        nstands = pools.shape[0]

        #allocate working variables
        age = np.zeros(nstands, dtype=np.int32)
        slowPools = np.zeros(nstands, dtype=np.float)
        spinup_state = np.zeros(nstands, dtype=np.uint32)
        rotation = np.zeros(nstands, dtype=np.int32)
        step = np.zeros(nstands, dtype=np.int32)
        lastRotationSlowC = np.zeros(nstands, dtype=np.float)
        disturbance_types = np.zeros(nstands, dtype=np.int32)
        enabled = np.ones(nstands, dtype=np.int32)

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
            #historic_disturbance_type, last_pass_disturbance_type,
            #    disturbance_types
            n_finished = self.dll.AdvanceSpinupState(
                spatial_unit, return_interval, min_rotations, max_rotations,
                inventory_age, delay, slowPools, historic_disturbance_type,
                last_pass_disturbance_type, afforestation_pre_type_id,
                spinup_state, disturbance_types, rotation, step,
                lastRotationSlowC, enabled)
            if n_finished == nstands:
                break
            logging.info("GetMerchVolumeGrowthOps")
            self.dll.GetMerchVolumeGrowthOps(ops["growth"],
                classifiers, pools, age, spatial_unit, None, None, None, None)

            logging.info("GetDisturbanceOps")
            self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit,
                                       disturbance_types)

            logging.info("ComputePools")
            self.dll.ComputePools([ops[x] for x in opSchedule], pools, enabled)

            self.dll.EndSpinupStep(spinup_state, pools,
                age, slowPools)
            if(debug):
                debug_output = debug_output.append(pd.DataFrame(data={
                    "index": list(range(nstands)),
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


    def step(self, pools, flux, classifiers, age, disturbance_types,
            spatial_unit, mean_annual_temp, transition_rule_ids,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled, land_class,
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
            "snag_turnover",
            "biomass_turnover",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        logging.info("AdvanceStandState")
        self.dll.AdvanceStandState(classifiers, spatial_unit,
            disturbance_types, transition_rule_ids, last_disturbance_type,
            time_since_last_disturbance, time_since_land_class_change,
            growth_enabled, enabled, land_class, regeneration_delay, age)

        logging.info("GetDisturbanceOps")
        self.dll.GetDisturbanceOps(ops["disturbance"], spatial_unit,
                                  disturbance_types)

        logging.info("GetMerchVolumeGrowthOps")
        self.dll.GetMerchVolumeGrowthOps(ops["growth"],
            classifiers, pools, age, spatial_unit, last_disturbance_type,
            time_since_last_disturbance, growth_multipliers, growth_enabled)
        
        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["snag_turnover"], ops["biomass_turnover"],
            spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"],
            ops["slow_mixing"], spatial_unit, mean_annual_temp)

        logging.info("Compute flux")
        self.dll.ComputeFlux([ops[x] for x in opSchedule],
            [self.opProcesses[x] for x in opSchedule], pools, flux, enabled)

        self.dll.EndStep(age, regeneration_delay)
        for x in self.opNames:
            self.dll.FreeOp(ops[x])
