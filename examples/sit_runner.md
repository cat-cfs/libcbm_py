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
import json, os
from types import SimpleNamespace
import notebook_startup
%matplotlib inline
```

```python
from libcbm import data_helpers
from libcbm.input.sit import sit_reader
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_simulator
```

```python
settings = notebook_startup.load_settings()
dll_path = settings["libcbm_path"]
db_path = settings["cbm_defaults_db_path"]
cbm_defaults_ref = CBMDefaultsReference(db_path)
```

```python
config_path = os.path.abspath("./sit/growth_only/sit_config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
sit_data = sit_reader.read(config["import_config"], os.path.dirname(config_path))
sit_mapping = SITMapping(config["mapping_config"], cbm_defaults_ref)
```

```python
cbm = sit_cbm_factory.initialize_cbm(
    db_path, dll_path, sit_data, sit_mapping)
```

```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(
    sit_data, sit_mapping)
```

```python
results = SimpleNamespace()
results.pool_indicators = None
results.flux_indicators = None
results.state_indicators = None

def append_simulation_result(timestep, cbm_vars):
    results.pool_indicators = data_helpers.append_simulation_result(
        results.pool_indicators, cbm_vars.pools, timestep)
    if timestep > 0:
        results.flux_indicators = data_helpers.append_simulation_result(
            results.flux_indicators, cbm_vars.flux_indicators, timestep)
    results.state_indicators = data_helpers.append_simulation_result(
        results.state_indicators, cbm_vars.state, timestep)
    
```

```python
cbm_simulator.simulate(
    cbm, 100, classifiers, inventory, cbm_defaults_ref.get_pools(), cbm_defaults_ref.get_flux_indicators(),
    pre_dynamics_func = lambda x: x, reporting_func=append_simulation_result)
```

```python
pi = results.pool_indicators
list(pi)
```

```python
biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots', 'SoftwoodFineRoots', 'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']

pi[pi.identifier==1][['timestep']+biomass_pools].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
list(results.state_indicators)
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
