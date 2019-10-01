---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Standard Import Tool Example

```python
import os, json
%matplotlib inline
```

Import the required packages from libcbm
 
    - sit_cbm_factory: a module for initializing the CBM model from the CBM Standard import tool format
    - cbm_simulator: simulates the sit dataset using the CBM model

```python
import notebook_startup
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
```

## Setup
Load the standard import tool configuration at the specified path.  This configuration encompasses the data source for the various sit inputs (sit_inventory, sit_classifiers etc.) and also the relationships of classifiers, and disturbance types to the default CBM parameters.

```python
config_path = os.path.abspath("./sit/growth_only/sit_config.json")
sit = sit_cbm_factory.load_sit(config_path)
```

Initialize and validate the inventory in the sit dataset

```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
```

Initialize an instance of the CBM model

```python
cbm = sit_cbm_factory.initialize_cbm(sit)
```

Create storage and a function for storing CBM simulation results.  This particular implementation appends timestep results for each step into a running DataFrame which is stored in memory.

```python
results, reporting_func = cbm_simulator.create_in_memory_reporting_func()
```

## Simulation
The following line of code spins up the CBM inventory and runs it through 100 timesteps. 

```python
cbm_simulator.simulate(
    cbm,
    n_steps              = 100,
    classifiers          = classifiers,
    inventory            = inventory,
    pool_codes           = sit.defaults.get_pools(),
    flux_indicator_codes = sit.defaults.get_flux_indicators(),
    pre_dynamics_func    = lambda x: x,
    reporting_func       = reporting_func
)
```

## Pool Results

```python
pi = results.pool_indicators
```

```python
pi.head()
```

```python
biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots', 'SoftwoodFineRoots',
                 'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']

dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
             'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
             'HardwoodStemSnag', 'HardwoodBranchSnag']

pi[pi.identifier==2][['timestep']+biomass_pools].groupby("timestep").sum().plot(figsize=(10,10))
pi[pi.identifier==2][['timestep']+dom_pools].groupby("timestep").sum().plot(figsize=(10,10))
```

## State Variable Results

```python
si = results.state_indicators
```

```python
si.head()
```

```python
state_variables = ['timestep','last_disturbance_type', 'time_since_last_disturbance', 'time_since_land_class_change',
 'growth_enabled', 'enabled', 'land_class', 'age', 'growth_multiplier', 'regeneration_delay']
si[si.identifier==1][state_variables].groupby('timestep').sum().plot(figsize=(10,10))
```

## Flux Indicators

```python
fi = results.flux_indicators
```

```python
fi.head()
```

```python
annual_process_fluxes = [
    'DecayDOMCO2Emission',
    'DeltaBiomass_AG',
    'DeltaBiomass_BG',
    'TurnoverMerchLitterInput',
    'TurnoverFolLitterInput',
    'TurnoverOthLitterInput',
    'TurnoverCoarseLitterInput',
    'TurnoverFineLitterInput',
    'DecayVFastAGToAir',
    'DecayVFastBGToAir',
    'DecayFastAGToAir',
    'DecayFastBGToAir',
    'DecayMediumToAir',
    'DecaySlowAGToAir',
    'DecaySlowBGToAir',
    'DecaySWStemSnagToAir',
    'DecaySWBranchSnagToAir',
    'DecayHWStemSnagToAir',
    'DecayHWBranchSnagToAir']

```

```python
fi[fi["identifier"]==1][["timestep"]+annual_process_fluxes].groupby("timestep").sum().plot(figsize=(15,10))
```

## Appendix


### SIT source data

```python
sit.sit_data.age_classes
```

```python
sit.sit_data.inventory
```

```python
sit.sit_data.classifiers
```

```python
sit.sit_data.classifier_values
```

```python
sit.sit_data.disturbance_types
```

```python
sit.sit_data.yield_table
```

```python
print(json.dumps(sit.config, indent=4, sort_keys=True))
```
