# Spatially Explicit Dataset Example

We had to patch the `libcbm` code to get it to run correctly with a "spatially explicit" SIT input dataset (which we exported from a spatially explicit Patchworks model of the UBC Malcolm Knapp Research Forest using the _Export to CBM-CFS3_ tool). 


See [this PDF document](https://carbon.nfis.org/cbm/downloadFile.action?file_id=1745) for details of "spatially explicit" (technically "spatially referenced" is a more accurate term for what is really happening here) modelling in CBM-CFS3 (requires NFIS login to access, concept maps well to `libcbm` which has the same structure as CBM-CFS3). The main `libcbm` developer (Scott Morken) confirmed that `libcbm` code was developed with the intent of accepting spatially explicit but that this path through the code was not throughly tested end-to-end with a full-sized dataset. Our patched code seems to run well, but testing is still in progress.

Our patched fork of `libcbm`is available here:

https://github.com/gparadis/libcbm_py

The patched fork will eventually be merged with the official distribution here:

https://github.com/cat-cfs/libcbm_py

We have added our "spatially explicit" test dataset (under `libcbm/resources/test/sit_spatially_explicit`) and this notebook (under `examples/sit_spatially_explicit`) to our patched fork. A version of this test dataset and notebook _may_ eventually be included in the official `libcbm` distribution. 

Import required packages and modules.


```python
import os, json
import pandas as pd
%matplotlib inline
```


```python
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm.storage.backends import BackendType
from libcbm import resources
```

## Setup
Load the standard import tool configuration at the specified path.  This configuration encompasses the data source for the various sit inputs (sit_inventory, sit_classifiers etc.) and also the relationships of classifiers, and disturbance types to the default CBM parameters.


```python
config_path = os.path.join(resources.get_test_resources_dir(), 
                           "sit_spatially_explicit", 
                           "sit_config.json")
sit = sit_cbm_factory.load_sit(config_path)
```

Initialize and validate the inventory in the SIT dataset.


```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
```

Create storage and a function for storing CBM simulation results.  This particular implementation appends timestep results for each step into a running DataFrame which is stored in memory.


```python
cbm_output = CBMOutput(
    classifier_map=sit.classifier_value_names,
    disturbance_type_map=sit.disturbance_name_map)
```

## Simulation

At this point the environment is ready to simulate growth and disturbance in each of our spatially referenced stands for a user-defined number of one-year discrete time steps (200 in the example below).


```python
with sit_cbm_factory.initialize_cbm(sit) as cbm:
    # Apply rule based disturbance events and transition rules based on the SIT input
    rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
    
    def pre_dynamics_func(t, cbm_vars):
        print(t)
        return rule_based_processor.pre_dynamics_func(t, cbm_vars)
    # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
    cbm_simulator.simulate(
        cbm,
        n_steps              = 200,
        classifiers          = classifiers,
        inventory            = inventory,
        pre_dynamics_func    = pre_dynamics_func,
        reporting_func       = cbm_output.append_simulation_result,
        #backend_type = BackendType.numpy
    )
```

Dump table of classifier values.


```python
cbm_output.classifiers.to_pandas()
```
## Results

### Pool Stocks


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

annual_carbon_stocks.groupby("Year").sum().plot(figsize=(10,10),xlim=(0,200),ylim=(0,5.4e6))

```
### State Variables


```python
si = cbm_output.state.to_pandas()
si.head()
```
```python
state_variables = ['timestep','last_disturbance_type', 'time_since_last_disturbance', 'time_since_land_class_change',
 'growth_enabled', 'enabled', 'land_class', 'age', 'growth_multiplier', 'regeneration_delay']
si[state_variables].groupby('timestep').mean().plot(figsize=(10,10))
```
### Pool Fluxes


```python
fi = cbm_output.flux.to_pandas()
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
### Disturbance Statistics

The following call returns `None`. This is expected, as the (patched) code for the spatially explicit case currently does not compile any disturbance statistics when a `RuleTargetResult` object instance is instantiated with the `spatially_indexed_target` function (i.e., the `statistics` attribute is explicitly set to `None`). 


```python
rule_based_processor.sit_event_stats_by_timestep[1]
```


```python
rule_based_processor.sit_events
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
