import os
from libcbmwrapper import LibCBMWrapper

os.chdir(r"C:\dev\LibCBM\LibCBM\x64\Debug")
w = LibCBMWrapper(dllpath=r"C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll")
w.Initialize(
    dbpath = r"C:\dev\LibCBM\LibCBM\LibCBM\cbm_defaults.db",
    random_seed = 1,
    classifiers = [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"}
    ],
    classifierValues = [
        {"id": 1, "classifier_id": 1, "name": "a1", "description": "a1"},
        {"id": 2, "classifier_id": 1, "name": "a2", "description": "a2"},
        {"id": 3, "classifier_id": 1, "name": "a3", "description": "a3"},
        {"id": 4, "classifier_id": 2, "name": "b1", "description": "b1"},
        {"id": 5, "classifier_id": 2, "name": "b2", "description": "b2"},
        {"id": 6, "classifier_id": 2, "name": "b3", "description": "b3"},
        {"id": 7, "classifier_id": 3, "name": "c1", "description": "c1"},
        {"id": 8, "classifier_id": 3, "name": "c2", "description": "c2"},
        {"id": 9, "classifier_id": 3, "name": "c3", "description": "c3"},
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

pools = w.Spinup(classifierSet=[1,4,7], 
                 spatial_unit_id = 1,
                 age = 100,
                 delay = 1,
                 historical_disturbance_type_id= 2,
                 last_pass_disturbance_type_id = 2)

result = w.Step(
    pools=pools,
    classifierSet=[1,4,7],
    age=100,
    spatial_unit_id=1,
    lastDisturbanceType=2,
    timeSinceLastDisturbance=0,
    disturbance_type_id=0,
    mean_annual_temp=None,
    get_flows=True)





