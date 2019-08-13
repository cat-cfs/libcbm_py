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
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.cbm.cbm_wrapper import CBMWrapper
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_variables
```

```python
dllpath = r'C:\dev\LibCBM\LibCBM_Build\build\LibCBM\Release\LibCBM.dll'
dbpath = 'C:\dev\cbm_defaults\cbm_defaults.db'

```

```python


ref = CBMDefaultsReference(dbpath)
#create a single classifier/classifier value for the single growth curve
classifiers_config = cbm_config.classifier_config([
    cbm_config.classifier("growth_curve", [
        cbm_config.classifier_value("growth_curve1")
    ])
])


transitions_config = []

merch_volume_to_biomass_config = cbm_config.merch_volume_to_biomass_config(
    dbpath, [
        cbm_config.merch_volume_curve(
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
handle_config = json.dumps(
        {
            "pools": pooldef,
            "flux_indicators": cbm_defaults.load_cbm_flux_indicators(dbpath)
        })
libcbm_handle = LibCBMHandle(dllpath,handle_config)

libcbm_wrapper = LibCBMWrapper(libcbm_handle)

cbm_config = json.dumps({
        "cbm_defaults": cbm_defaults.load_cbm_parameters(dbpath),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    })
cbm_wrapper = CBMWrapper(libcbm_handle, cbm_config)

```

## Set up the simulation variables  ##


```python
classifiers = pd.DataFrame({"growth_curve": np.ones(1, dtype=np.int32)})
inventory = pd.DataFrame({
    "age": [0],
    "spatial_unit": [42],
    "afforestation_pre_type_id": [0],
    "land_class": [1],
    "historical_disturbance_type": [0],
    "last_pass_disturbance_type": [0],
    "delay": [0],
})

inventory = cbm_variables.initialize_inventory(classifiers, inventory)
n_stands = inventory.age.shape[0]
state = cbm_variables.initialize_cbm_state_variables(n_stands)
#ensure growth is enabled
state.growth_enabled = 1
pools = cbm_variables.initialize_pools(n_stands, ref.get_pools())

```

### Run the simulation for several iterations ###
This includes:
 1. computing the merchantable growth operation matrix 
 2. applying the growth operation matrix to the pools
 3. save the result to a pandas dataframe
 4. increment the age
 

```python

op = libcbm_wrapper.AllocateOp(n_stands)

result = None
for i in range(0, 200):
    cbm_wrapper.GetMerchVolumeGrowthOps(
        op, inventory, pools, state)
     
    # note the duplication of op here, CBM applies the growth operation 2 times per timestep
    # so this is for consistency
    libcbm_wrapper.ComputePools([op, op], pools)
   

    result = cbm_variables.append_simulation_result(result, pools, i+1) 
    
    state.age += 1
    
result = result.reset_index(drop=True)
```

```python
result \
    [['SoftwoodMerch','SoftwoodFoliage','SoftwoodOther','SoftwoodCoarseRoots','SoftwoodFineRoots']] \
    .plot(figsize=(10,10), kind="area")
```

```python

```
