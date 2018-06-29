import os, json
import numpy as np
from cbm_defaults import load_cbm_parameters

from libcbmwrapper import LibCBMWrapper

os.chdir(r"C:\dev\LibCBM\LibCBM\x64\Debug")
w = LibCBMWrapper(dllpath=r"C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll")

dbpath = r"C:\dev\LibCBM\cbm_defaults.db"

configuration = {
    "pools": [
        "Input",
        "SoftwoodMerch",
        "SoftwoodFoliage",
        "SoftwoodOther",
        "SoftwoodCoarseRoots",
        "SoftwoodFineRoots",
        "HardwoodMerch",
        "HardwoodFoliage",
        "HardwoodOther",
        "HardwoodCoarseRoots",
        "HardwoodFineRoots",
        "AboveGroundVeryFastSoil",
        "BelowGroundVeryFastSoil",
        "AboveGroundFastSoil",
        "BelowGroundFastSoil",
        "MediumSoil",
        "AboveGroundSlowSoil",
        "BelowGroundSlowSoil",
        "SoftwoodStemSnag",
        "SoftwoodBranchSnag",
        "HardwoodStemSnag",
        "HardwoodBranchSnag",
        "CO2",
        "CH4",
        "CO",
        "Products",
        ],
    "classifiers": [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"}
        ],
    "classifier_values": [
        {"id": 1, "classifier_id": 1, "value": "a1", "description": "a1"},
        {"id": 2, "classifier_id": 1, "value": "a2", "description": "a2"},
        {"id": 3, "classifier_id": 1, "value": "a3", "description": "a3"},
        {"id": 4, "classifier_id": 2, "value": "b1", "description": "b1"},
        {"id": 5, "classifier_id": 2, "value": "b2", "description": "b2"},
        {"id": 6, "classifier_id": 2, "value": "b3", "description": "b3"},
        {"id": 7, "classifier_id": 3, "value": "c1", "description": "c1"},
        {"id": 8, "classifier_id": 3, "value": "c2", "description": "c2"},
        {"id": 9, "classifier_id": 3, "value": "c3", "description": "c3"},
    ],
    "merch_volume_to_biomass": {
        "db_path": dbpath,
        "merch_volume_curves": [
            {
                "classifier_set": { 
                    "type": "name",
                    "values": ["a1","b1", "?"]
                },
                "softwood_component": {
                    "species_id": 10,
                    "age_volume_pairs":[[0,0.00],[5,0.01],[20,20],[50,100]]
                }
            }
        ]
    }
}
configuration["cbm_defaults"] = load_cbm_parameters(dbpath)

configString = json.dumps(configuration)#, ensure_ascii=True)
w.Initialize(configString)
pools = np.zeros((2,27), dtype=np.double)
pools[:,0] = 1.0
classifiers = np.array([[1,4,7],[1,4,7]],dtype=np.int32)
ages = np.array([0,0], dtype=np.int32)
spus = np.array([3,3], dtype=np.int32)
last_dist = np.array([0,0], dtype=np.int32)
dist = np.array([1,1], dtype=np.int32)
growth_mult = np.array([1.0,1.0], dtype=np.double)

returnInterval = np.array([125,125], dtype=np.int32)
minRotations = np.array([10,10], dtype = np.int32)
maxRotations = np.array([30,30], dtype = np.int32)
finalAge = np.array([300,100], dtype = np.int32)
delay = np.array([0,1], dtype = np.int32)
slowPools= np.array([0,0], dtype = np.double)
state= np.array([0,0], dtype = np.uint32)
rotation= np.array([0,0], dtype = np.int32)
step= np.array([0,0], dtype = np.int32)
lastRotationSlowC= np.array([0.0,0.0], dtype = np.double)

saved_pools = None
for i in range(300):
    #print(i)
    #w.AdvanceSpinupState(returnInterval, minRotations, maxRotations,
    #                        finalAge, delay, slowPools, state, rotation, step,
    #                        lastRotationSlowC)

    #print(step)
    op1 = w.GetMerchVolumeGrowthAndDeclineOps(
        classifiers, pools, ages, spus, last_dist, last_dist, growth_mult)

    #op2 = w.GetTurnoverOps(spus)
    #op3 = w.GetDecayOps(spus)
    #op4 = w.GetDisturbanceOps(spus, dist)
    w.ComputePools([op1[0]["id"]], pools)
    ages = ages + 1
    if saved_pools is None:
        saved_pools = np.matrix(pools[1,0:12])
    else:
        saved_pools = np.vstack((saved_pools, np.matrix(pools[1,0:12])))

np.savetxt("out.csv", saved_pools, delimiter=",")






