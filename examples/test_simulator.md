---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
import notebook_startup
```

```python
from libcbm.test.cbm import case_generation
from libcbm.test.cbm import test_case_simulator
```

```python
settings = notebook_startup.load_settings()
cbm_defaults_db_path = settings["cbm_defaults_db_path"]
libcbm_path = settings["libcbm_path"]
```

```python
case = {
    "id": 1,
    "age": 20,
    "area": 1,
    "delay": 0,
    "afforestation_pre_type": None,
    "unfccc_land_class": "UNFCCC_FL_R_FL",
    "admin_boundary": "British Columbia",
    "eco_boundary": "Pacific Maritime",
    "historical_disturbance": "Wildfire",
    "last_pass_disturbance": "Wildfire",
    "components": [
        {
            "species": "Spruce",
            "age_volume_pairs": [(0,0),(10,20),(50,100),(100,200),(300,300)]
        }
    ],
    "events": [
        {"disturbance_type": "Deforestation", "time_step": 60},
        {"disturbance_type": "Wildfire", "time_step": 20}
    ]}
```

```python
result = test_case_simulator.run_test_cases(
    dll_path=libcbm_path,
    db_path=cbm_defaults_db_path,
    cases=[case],
    n_steps=100)
```

```python
pools = result["pools"]
pools[["timestep",'SoftwoodMerch','SoftwoodFoliage','SoftwoodOther',
       'SoftwoodCoarseRoots','SoftwoodFineRoots','HardwoodMerch',
       'HardwoodFoliage','HardwoodOther','HardwoodCoarseRoots',
       'HardwoodFineRoots',]].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
pools[["timestep",'AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil',
       'AboveGroundFastSoil', 'BelowGroundFastSoil', 'MediumSoil',
       'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag',
       'SoftwoodBranchSnag', 'HardwoodStemSnag', 'HardwoodBranchSnag']] \
    .groupby("timestep").sum().plot(figsize=(10,10))
```

```python
state_variables = result["state"]
state_variables[['timestep', 'age', 'land_class', 'last_disturbance_type', 
                 'time_since_last_disturbance', 'time_since_land_class_change',
                 'growth_enabled', 'growth_multiplier', 'regeneration_delay',
                 'enabled']].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
flux = result['flux']
flux[['timestep', 'DecayDOMCO2Emission', 'DecayFastAGToAir', 'DecayFastBGToAir',
      'DecayHWBranchSnagToAir', 'DecayHWStemSnagToAir', 'DecayMediumToAir',
      'DecaySWBranchSnagToAir', 'DecaySWStemSnagToAir', 'DecaySlowAGToAir',
      'DecaySlowBGToAir', 'DecayVFastAGToAir', 'DecayVFastBGToAir']] \
    .groupby("timestep").sum().plot(figsize=(10,10))
```

```python
flux = result['flux']
flux[['timestep', 'TurnoverCoarseLitterInput', 'TurnoverFineLitterInput', 'TurnoverFolLitterInput',
 'TurnoverMerchLitterInput', 'TurnoverOthLitterInput', 'DeltaBiomass_AG', 'DeltaBiomass_BG',]] \
    .groupby("timestep").sum().plot(figsize=(10,10))
```

```python
flux = result['flux']
flux[['DisturbanceBioCH4Emission', 'DisturbanceBioCO2Emission', 
      'DisturbanceBioCOEmission', 'DisturbanceCH4Production', 'DisturbanceCO2Production',
      'DisturbanceCOProduction', 'DisturbanceCoarseLitterInput', 'DisturbanceCoarseToAir',
      'DisturbanceDOMCH4Emssion', 'DisturbanceDOMCO2Emission', 'DisturbanceDOMCOEmission',
      'DisturbanceDOMProduction', 'DisturbanceFastAGToAir', 'DisturbanceFastBGToAir',
      'DisturbanceFineLitterInput', 'DisturbanceFineToAir', 'DisturbanceFolLitterInput',
      'DisturbanceFolToAir', 'DisturbanceHWBranchSnagToAir', 'DisturbanceHWStemSnagToAir',
      'DisturbanceHardProduction', 'DisturbanceMediumToAir', 'DisturbanceMerchLitterInput',
      'DisturbanceMerchToAir', 'DisturbanceOthLitterInput', 'DisturbanceOthToAir',
      'DisturbanceSWBranchSnagToAir', 'DisturbanceSWStemSnagToAir', 'DisturbanceSlowAGToAir',
      'DisturbanceSlowBGToAir', 'DisturbanceSoftProduction', 'DisturbanceVFastAGToAir',
      'DisturbanceVFastBGToAir']].sum().plot(kind="bar",figsize=(10,10))
```
