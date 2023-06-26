---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Example for varying mean annual temperature in the CBM-SIT workflow

```python
import os, json
import pandas as pd
import numpy as np
%matplotlib inline
```

```python
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm.storage.backends import BackendType
from libcbm import resources
from libcbm.model.cbm.cbm_temperature_processor import (
    SpatialUnitMeanAnnualTemperatureProcessor,
)
```

```python
config_path = os.path.join(resources.get_test_resources_dir(), "cbm3_tutorial2", "sit_config.json")
sit = sit_cbm_factory.load_sit(config_path)
```

```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
```

```python
cbm_output = CBMOutput(
    classifier_map=sit.classifier_value_names,
    disturbance_type_map=sit.disturbance_name_map)
```

```python
# example mean annual temperature table
mean_annual_temperature_data = pd.DataFrame(
    {
        # by convention in CBM3 t=0 refers to spinup
        "timestep": np.arange(0,201),
        # note only one spatial unit is in use for this example
        "spatial_unit": 19,
        # generating a simple ramp here, but it's also easy to inform this information based on a file such as csv
        "mean_annual_temp": np.arange(0, 2+2/200, 2/200)
    }
)
mean_annual_temp_processor = SpatialUnitMeanAnnualTemperatureProcessor(mean_annual_temperature_data)
```

```python
mean_annual_temperature_data
```

## Simulation


```python
with sit_cbm_factory.initialize_cbm(sit) as cbm:

    rule_based_disturbance_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)

    def pre_dynamics_func(timestep, cbm_vars):
        # this function applies the rule based disturbances and sets the mean annual temperature 
        cbm_vars = rule_based_disturbance_processor.pre_dynamics_func(timestep, cbm_vars)
        cbm_vars = mean_annual_temp_processor.set_timestep_mean_annual_temperature(timestep, cbm_vars)
        return cbm_vars

    # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
    cbm_simulator.simulate(
        cbm,
        n_steps = 200,
        classifiers = classifiers,
        inventory = inventory,
        pre_dynamics_func = pre_dynamics_func,
        reporting_func = cbm_output.append_simulation_result,
        spinup_params = mean_annual_temp_processor.get_spinup_parameters(inventory),
        backend_type = BackendType.numpy
    )
```

```python
cbm_output.classifiers.to_pandas()
```

```python
cbm_output.parameters.to_pandas()[["timestep", "mean_annual_temp"]].groupby("timestep").mean().plot()
```

## Pool Results

```python
pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), left_on=["identifier", "timestep"], right_on=["identifier", "timestep"])
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

biomass_result = pi[['timestep']+biomass_pools]
dom_result = pi[['timestep']+dom_pools]
total_eco_result = pi[['timestep']+biomass_pools+dom_pools]

annual_carbon_stocks = pd.DataFrame(
    {
        "Year": pi["timestep"],
        "Biomass": pi[biomass_pools].sum(axis=1),
        "DOM": pi[dom_pools].sum(axis=1),
        "Total Ecosystem": pi[biomass_pools+dom_pools].sum(axis=1)})

annual_carbon_stocks.groupby("Year").sum().plot(figsize=(10,10),xlim=(0,160),ylim=(0,5.4e6))
```

```python
annual_carbon_stocks.groupby("Year").sum()
```

## State Variable Results

```python
si = cbm_output.state.to_pandas()
```

```python
si.head()
```

```python
state_variables = ['timestep', 'time_since_last_disturbance', 'time_since_land_class_change',
 'growth_enabled', 'enabled', 'land_class', 'age', 'growth_multiplier', 'regeneration_delay']
si[state_variables].groupby('timestep').mean().plot(figsize=(10,10))
```

## Flux Indicators

```python
fi = cbm_output.flux.to_pandas()
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
fi[["timestep"]+annual_process_fluxes].groupby("timestep").sum().plot(figsize=(15,10))
```

## Disturbance Statistics

```python
rule_based_disturbance_processor.sit_event_stats_by_timestep[100]
```

```python
rule_based_disturbance_processor.sit_events
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
sit.sit_data.disturbance_events
```

```python
sit.sit_data.transition_rules
```

```python
sit.sit_data.yield_table
```

```python
print(json.dumps(sit.config, indent=4, sort_keys=True))
```
