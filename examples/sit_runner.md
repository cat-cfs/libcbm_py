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

# Standard Import tool demonstration

```python
import os
%matplotlib inline
```

```python
import notebook_startup
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
```

```python
config_path = os.path.abspath("./sit/growth_only/sit_config.json")
sit = sit_cbm_factory.load_sit(config_path)
```

```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
```

```python
cbm = sit_cbm_factory.initialize_cbm(sit)
```

```python
results, reporting_func = cbm_simulator.create_in_memory_reporting_func()
```

```python
cbm_simulator.simulate(
    cbm, 100, classifiers, inventory, 
    sit.defaults.get_pools(), 
    sit.defaults.get_flux_indicators(), 
    pre_dynamics_func = lambda x: x,
    reporting_func=reporting_func)

```

```python

```

```python

```

```python
pi = results.pool_indicators
```

```python
biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots', 'SoftwoodFineRoots',
                 'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']

dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
             'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
             'HardwoodStemSnag', 'HardwoodBranchSnag']

pi[pi.identifier==1][['timestep']+biomass_pools].groupby("timestep").sum().plot(figsize=(10,10))
pi[pi.identifier==1][['timestep']+dom_pools].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
list(results.state_indicators)
si = results.state_indicators
```

```python
state_variables = ['timestep','last_disturbance_type', 'time_since_last_disturbance', 'time_since_land_class_change',
 'growth_enabled', 'enabled', 'land_class', 'age', 'growth_multiplier', 'regeneration_delay']
si[si.identifier==1][state_variables].groupby('timestep').sum().plot(figsize=(10,10))
```

```python
fi = results.flux_indicators
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

```python

```

```python

```
