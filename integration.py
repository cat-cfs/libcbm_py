import os
from libcbmwrapper import LibCBMWrapper
import numpy as np

os.chdir(r"C:\dev\LibCBM\LibCBM\x64\Debug")
w = LibCBMWrapper(dllpath=r"C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll")
w.Initialize(
    dbpath = b"C:\dev\LibCBM\LibCBM\LibCBM\cbm_defaults.db",
    random_seed = 1,
    classifiers = [
        {"id": 1, "name": b"a"},
        {"id": 2, "name": b"b"},
        {"id": 3, "name": b"c"}
    ],
    classifierValues = [
        {"id": 1, "classifier_id": 1, "name": b"a1", "description": b"a1"},
        {"id": 2, "classifier_id": 1, "name": b"a2", "description": b"a2"},
        {"id": 3, "classifier_id": 1, "name": b"a3", "description": b"a3"},
        {"id": 4, "classifier_id": 2, "name": b"b1", "description": b"b1"},
        {"id": 5, "classifier_id": 2, "name": b"b2", "description": b"b2"},
        {"id": 6, "classifier_id": 2, "name": b"b3", "description": b"b3"},
        {"id": 7, "classifier_id": 3, "name": b"c1", "description": b"c1"},
        {"id": 8, "classifier_id": 3, "name": b"c2", "description": b"c2"},
        {"id": 9, "classifier_id": 3, "name": b"c3", "description": b"c3"},
    ],
    merchVolumeCurves = [
        {"classifier_value_ids": [1,4,7],
         "sw_component": {
             "species_id": 14,
             "ages": [10,20,30,40],
             "volumes": [10,20,30,40]
          },
         "hw_component": {
             "species_id": 14,
             "ages": [10,20,30,40],
             "volumes": [10,20,30,40]
          },
         }]
    )
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






