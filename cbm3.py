from libcbmwrapper import LibCBM_SpinupState
import cbm_defaults
import numpy as np
import json, os,logging
class CBM3:
    def __init__(self, dll, dbpath):
        self.dll = dll
        self.dbpath = dbpath

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

    def initialize(self, config):
        self.dll.Initialize(config)

    def initialize_config(self, merch_volume_to_biomass, classifiers, transitions, save_path=None):
        '''
        initialize config sets up the json configuration object passed to the underlying dll
        returns config as string, and optionally saves to specified path
        '''
        self.configuration = {}
        self.configuration["cbm_defaults"] = cbm_defaults.load_cbm_parameters(self.dbpath)
        self.configuration["pools"] = cbm_defaults.load_cbm_pools(self.dbpath)
        self.configuration["flux_indicators"] = cbm_defaults.load_flux_indicators(self.dbpath)
        self.configuration["merch_volume_to_biomass"] = merch_volume_to_biomass
        self.configuration["classifiers"] = classifiers["classifiers"]
        self.configuration["classifier_values"] = classifiers["classifier_values"]
        self.configuration["transitions"] = transitions
        configString = json.dumps(self.configuration, indent=4)#, ensure_ascii=True)
        if save_path:
            with open(save_path, 'w') as configfile:
                configfile.write(configString)
        return configString


    def load_config(self, path):
        '''
        loads configuration json from an exisiting file
        '''
        with open(path, 'r') as configfile:
            configString = configfile.read()
            self.configuration = json.loads(configString)
        return configString

    def promoteScalar(self, value, size, dtype):
        '''
        if the specified value is scalar promote it to a numpy array filled with the scalar value
        otherwise return the value
        '''
        if isinstance(value, np.ndarray):
            return value
        else:
            return np.ones(size, dtype=dtype) * value

    def spinup(self, pools, classifiers, inventory_age, spatial_unit,
               historic_disturbance_type, last_pass_disturbance_type,
              return_interval, min_rotations, max_rotations, delay,
              debug_output_path = None):
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

        spatial_unit = self.promoteScalar(spatial_unit, nstands, dtype=np.int32)
        historic_disturbance_type = self.promoteScalar(historic_disturbance_type, nstands, dtype=np.int32)
        last_pass_disturbance_type = self.promoteScalar(last_pass_disturbance_type, nstands, dtype=np.int32)
        return_interval = self.promoteScalar(return_interval, nstands, dtype=np.int32)
        min_rotations = self.promoteScalar(min_rotations, nstands, dtype=np.int32)
        max_rotations = self.promoteScalar(max_rotations, nstands, dtype=np.int32)
        delay = self.promoteScalar(delay, nstands, dtype=np.int32)

        logging.info("AllocateOp")
        ops = { x: self.dll.AllocateOp(nstands) for x in self.opNames }

        logging.info("GetTurnoverOps")
        self.dll.GetTurnoverOps(ops["biomass_turnover"], ops["snag_turnover"], spatial_unit)

        logging.info("GetDecayOps")
        self.dll.GetDecayOps(ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"], spatial_unit)

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

        if debug_output_path:
            with open(debug_output_path, 'w') as debug_file:
                debug_file.write(
                    ",".join(
                    ["id", 
                     ",".join([x["name"] for x in self.configuration["classifiers"]]),
                     ",".join([x["name"] for x in self.configuration["pools"]]),
                     "inventory_age", "spatial_unit", "historic_disturbance_type",
                     "last_pass_disturbance_type", "return_interval", "min_rotations",
                     "max_rotations", "delay", "age", "slowPools", "spinup_state",
                     "rotation", "step", "lastRotationSlowC", "disturbance_types"]) + os.linesep)

        while (True):
            logging.info("AdvanceSpinupState")
            n_finished = self.dll.AdvanceSpinupState(return_interval, min_rotations, max_rotations, inventory_age, delay, 
                                    slowPools, spinup_state, rotation, step, lastRotationSlowC)
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

            if debug_output_path:
                with open(debug_output_path, 'ab') as debug_file:
                    iteration_data = np.column_stack((
                        np.arange(1, classifiers.shape[0]+1),
                        classifiers, pools, inventory_age, spatial_unit,
                        historic_disturbance_type, last_pass_disturbance_type,
                        return_interval, min_rotations, max_rotations, delay,
                        age, slowPools, spinup_state, rotation, step,
                        lastRotationSlowC, disturbance_types))
                    np.savetxt(debug_file, iteration_data, delimiter=",")

        for x in self.opNames:
            self.dll.FreeOp(ops[x])
