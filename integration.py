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
classifiers = np.array([[1,4,7],[1,4,7]],dtype=np.int32)
ages = np.array([0,0], dtype=np.int32)
spus = np.array([3,3], dtype=np.int32)
last_dist = np.array([0,0], dtype=np.int32)
w.GetMerchVolumeGrowthAndDeclineOps(
    classifiers, pools, ages, spus, last_dist, last_dist)

#pools = w.Spinup(classifierSet=[1,4,7], 
#                 spatial_unit_id = 1,
#                 age = 100,
#                 delay = 1,
#                 historical_disturbance_type_id= 2,
#                 last_pass_disturbance_type_id = 2)

#result = w.Step(
#    pools=pools,
#    classifierSet=[1,4,7],
#    age=100,
#    spatial_unit_id=1,
#    lastDisturbanceType=2,
#    timeSinceLastDisturbance=0,
#    disturbance_type_id=0,
#    mean_annual_temp=None,
#    get_flows=True)





