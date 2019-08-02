---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os, json
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
from libcbm.wrapper.libcbmwrapper import LibCBMWrapper
from libcbm.configuration import cbmconfig
from libcbm.configuration import cbm_defaults
from libcbm.configuration.cbm_defaults_reference import CBMDefaultsReference
```

```python
dllpath = r'C:\dev\LibCBM\LibCBM_Build\build\LibCBM\Release\LibCBM.dll'
dbpath = 'C:\dev\cbm_defaults\cbm_defaults.db'
dll = LibCBMWrapper(dllpath)
```

```python


ref = CBMDefaultsReference(dbpath)
#create a single classifier/classifier value for the single growth curve
classifiers_config = cbmconfig.classifier_config([
    cbmconfig.classifier("growth_curve", [
        cbmconfig.classifier_value("growth_curve1")
    ])
])


transitions_config = []

merch_volume_to_biomass_config = cbmconfig.merch_volume_to_biomass_config(
    dbpath, [
        cbmconfig.merch_volume_curve(
            classifier_set = ['growth_curve1'], 
            merch_volumes = [
                {
                    "species_id": ref.get_species_id("Spruce"), 
                    "age_volume_pairs":[
                    (0, 0.0),
                    (10, 10.0),
                    (20, 15.0),
                    (30, 20.0),
                    (40, 25.0),
                    (50, 30.0),
                    (60, 35.0),
                    (70, 40.0),
                    (80, 45.0),
                    (90, 50.0),
                    (100, 55.0),
                    (110, 60.0),
                    (120, 65.0),
                    (130, 70.0),
                    (140, 75.0),
                    (150, 80.0),
                    (160, 85.625),
                    (170, 90.73529412),
                    (180, 95.84558824),
                    (190, 100.9558824)]}]
    )] )


```

```python
pooldef = cbm_defaults.load_cbm_pools(dbpath)
dll.Initialize(json.dumps(
        {
            "pools": pooldef,
            "flux_indicators": cbm_defaults.load_flux_indicators(dbpath)
        }))


cbm_config = {
        "cbm_defaults": cbm_defaults.load_cbm_parameters(dbpath),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }
dll.InitializeCBM(json.dumps(cbm_config))

```

## Set up the simulation variables  ##


```python
nstands = 1
age = np.array([0],dtype=np.int32)
classifiers = np.array([1], dtype=np.int32)
spatial_units = np.array([42],dtype=np.int32)
pools = np.zeros((1,len(pooldef)))
pools[:,0] = 1.0

```

### Run the simulation for several iterations ###
This includes:
 1. computing the merchantable growth operation matrix 
 2. applying the growth operation matrix to the pools
 3. save the result to a pandas dataframe
 4. increment the age
 

```python

op = dll.AllocateOp(nstands)

result = pd.DataFrame()
for i in range(0, 200):
    dll.GetMerchVolumeGrowthOps(
        op, 
        classifiers=classifiers,
        pools=pools,
        ages=age,
        spatial_units=spatial_units,
        last_dist_type=None,
        time_since_last_dist=None,
        growth_multipliers=None,
        growth_enabled=True)
     
    # note the duplication of op here, CBM applies the growth operation 2 times per timestep
    # so this is for consistency
    dll.ComputePools([op, op], pools)
    
    iteration_result = pd.DataFrame({x["name"]: pools[:,x["index"]] for x in pooldef})
    iteration_result.reset_index(level=0, inplace=True)
    result = result.append(iteration_result)
    
    age += 1
    
result = result.reset_index(drop=True)
```

```python
result \
    [['SoftwoodMerch','SoftwoodFoliage','SoftwoodOther','SoftwoodCoarseRoots','SoftwoodFineRoots']] \
    .plot(figsize=(10,10), kind="area")
```

```python

```
