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
results, reporting_func = cbm_simulator.create_in_memory_reporting_func(
    classifier_map=sit.classifier_value_names,
    disturbance_type_map=sit.disturbance_name_map)
```

## Simulation

At this point the environment is ready to simulate growth and disturbance in each of our spatially referenced stands for a user-defined number of one-year discrete time steps (200 in the example below).


```python
with sit_cbm_factory.initialize_cbm(sit) as cbm:
    # Apply rule based disturbance events and transition rules based on the SIT input
    rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
    # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
    cbm_simulator.simulate(
        cbm,
        n_steps              = 200,
        classifiers          = classifiers,
        inventory            = inventory,
        pre_dynamics_func    = rule_based_processor.pre_dynamics_func,
        reporting_func       = reporting_func
    )
```

Dump table of classifier values.


```python
results.classifiers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>timestep</th>
      <th>Track</th>
      <th>State</th>
      <th>au</th>
      <th>LeadSpecies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>managed</td>
      <td>2121</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1415</th>
      <td>1416</td>
      <td>200</td>
      <td>8</td>
      <td>managed</td>
      <td>2022</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>1416</th>
      <td>1417</td>
      <td>200</td>
      <td>8</td>
      <td>managed</td>
      <td>2022</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>1417</th>
      <td>1418</td>
      <td>200</td>
      <td>4</td>
      <td>managed</td>
      <td>2004</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>1419</td>
      <td>200</td>
      <td>12</td>
      <td>managed</td>
      <td>2023</td>
      <td>softwood</td>
    </tr>
    <tr>
      <th>1419</th>
      <td>1420</td>
      <td>200</td>
      <td>100</td>
      <td>managed</td>
      <td>2002</td>
      <td>softwood</td>
    </tr>
  </tbody>
</table>
<p>285420 rows × 6 columns</p>
</div>



## Results

### Pool Stocks


```python
pi = results.classifiers.merge(results.pools, left_on=["identifier", "timestep"], right_on=["identifier", "timestep"])
```


```python
pi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>timestep</th>
      <th>Track</th>
      <th>State</th>
      <th>au</th>
      <th>LeadSpecies</th>
      <th>Input</th>
      <th>SoftwoodMerch</th>
      <th>SoftwoodFoliage</th>
      <th>SoftwoodOther</th>
      <th>...</th>
      <th>BelowGroundSlowSoil</th>
      <th>SoftwoodStemSnag</th>
      <th>SoftwoodBranchSnag</th>
      <th>HardwoodStemSnag</th>
      <th>HardwoodBranchSnag</th>
      <th>CO2</th>
      <th>CH4</th>
      <th>CO</th>
      <th>NO2</th>
      <th>Products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>managed</td>
      <td>2121</td>
      <td>softwood</td>
      <td>2.045750</td>
      <td>151.416222</td>
      <td>21.578364</td>
      <td>82.055402</td>
      <td>...</td>
      <td>420.769253</td>
      <td>11.511631</td>
      <td>5.107295</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55383.856357</td>
      <td>17.667886</td>
      <td>159.015723</td>
      <td>0.0</td>
      <td>346.803167</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>7.437000</td>
      <td>7.808557</td>
      <td>26.382624</td>
      <td>13.496410</td>
      <td>...</td>
      <td>2393.902879</td>
      <td>0.074194</td>
      <td>0.117392</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296501.104751</td>
      <td>83.850936</td>
      <td>754.682807</td>
      <td>0.0</td>
      <td>1966.618771</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>0.118866</td>
      <td>0.124805</td>
      <td>0.421675</td>
      <td>0.215714</td>
      <td>...</td>
      <td>38.261888</td>
      <td>0.001186</td>
      <td>0.001876</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4738.994261</td>
      <td>1.340194</td>
      <td>12.062139</td>
      <td>0.0</td>
      <td>31.432581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
      <td>7.194100</td>
      <td>3.400270</td>
      <td>25.130360</td>
      <td>0.894678</td>
      <td>...</td>
      <td>2266.937495</td>
      <td>0.034053</td>
      <td>0.004243</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>279331.838066</td>
      <td>83.452469</td>
      <td>751.096213</td>
      <td>0.0</td>
      <td>2047.958936</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
      <td>4.298460</td>
      <td>3.051019</td>
      <td>16.765435</td>
      <td>3.701929</td>
      <td>...</td>
      <td>1355.137148</td>
      <td>0.034475</td>
      <td>0.022256</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>166961.314190</td>
      <td>49.862679</td>
      <td>448.778447</td>
      <td>0.0</td>
      <td>1223.651266</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4281f20b20>




    
![png](output_18_1.png)
    


### State Variables


```python
si = results.state
si.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>timestep</th>
      <th>last_disturbance_type</th>
      <th>time_since_last_disturbance</th>
      <th>time_since_land_class_change</th>
      <th>growth_enabled</th>
      <th>enabled</th>
      <th>land_class</th>
      <th>age</th>
      <th>growth_multiplier</th>
      <th>regeneration_delay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>CC</td>
      <td>69</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>69</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>CC</td>
      <td>6</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>CC</td>
      <td>6</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>CC</td>
      <td>7</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>CC</td>
      <td>8</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_variables = ['timestep','last_disturbance_type', 'time_since_last_disturbance', 'time_since_land_class_change',
 'growth_enabled', 'enabled', 'land_class', 'age', 'growth_multiplier', 'regeneration_delay']
si[state_variables].groupby('timestep').mean().plot(figsize=(10,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4281e7aa90>




    
![png](output_21_1.png)
    


### Pool Fluxes


```python
fi = results.flux
fi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>timestep</th>
      <th>DisturbanceCO2Production</th>
      <th>DisturbanceCH4Production</th>
      <th>DisturbanceCOProduction</th>
      <th>DisturbanceBioCO2Emission</th>
      <th>DisturbanceBioCH4Emission</th>
      <th>DisturbanceBioCOEmission</th>
      <th>DecayDOMCO2Emission</th>
      <th>DisturbanceSoftProduction</th>
      <th>...</th>
      <th>DisturbanceVFastBGToAir</th>
      <th>DisturbanceFastAGToAir</th>
      <th>DisturbanceFastBGToAir</th>
      <th>DisturbanceMediumToAir</th>
      <th>DisturbanceSlowAGToAir</th>
      <th>DisturbanceSlowBGToAir</th>
      <th>DisturbanceSWStemSnagToAir</th>
      <th>DisturbanceSWBranchSnagToAir</th>
      <th>DisturbanceHWStemSnagToAir</th>
      <th>DisturbanceHWBranchSnagToAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>




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




    <matplotlib.axes._subplots.AxesSubplot at 0x7f42832b8820>




    
![png](output_25_1.png)
    


### Disturbance Statistics

The following call returns `None`. This is expected, as the (patched) code for the spatially explicit case currently does not compile any disturbance statistics when a `RuleTargetResult` object instance is instantiated with the `spatially_indexed_target` function (i.e., the `statistics` attribute is explicitly set to `None`). 


```python
rule_based_processor.sit_event_stats_by_timestep[1]
```


```python
rule_based_processor.sit_events
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Track</th>
      <th>State</th>
      <th>au</th>
      <th>LeadSpecies</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>MinYearsSinceDist</th>
      <th>MaxYearsSinceDist</th>
      <th>LastDistTypeID</th>
      <th>MinTotBiomassC</th>
      <th>...</th>
      <th>MaxHWMerchStemSnagC</th>
      <th>efficiency</th>
      <th>sort_type</th>
      <th>target_type</th>
      <th>target</th>
      <th>disturbance_type</th>
      <th>time_step</th>
      <th>spatial_reference</th>
      <th>disturbance_type_id</th>
      <th>sort_field</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>managed</td>
      <td>2121</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC</td>
      <td>30</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>managed</td>
      <td>2021</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>95</td>
      <td>14</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>managed</td>
      <td>2021</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>178</td>
      <td>14</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>managed</td>
      <td>2021</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>260</td>
      <td>14</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC</td>
      <td>60</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4633</th>
      <td>430</td>
      <td>managed</td>
      <td>451</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC</td>
      <td>162</td>
      <td>1868</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4634</th>
      <td>136</td>
      <td>managed</td>
      <td>2102</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC</td>
      <td>40</td>
      <td>1869</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4635</th>
      <td>100</td>
      <td>managed</td>
      <td>2002</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>116</td>
      <td>1869</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4636</th>
      <td>100</td>
      <td>managed</td>
      <td>2002</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>188</td>
      <td>1869</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4637</th>
      <td>100</td>
      <td>managed</td>
      <td>2002</td>
      <td>softwood</td>
      <td>1</td>
      <td>999</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>SVOID</td>
      <td>Proportion</td>
      <td>1</td>
      <td>CC$1</td>
      <td>279</td>
      <td>1869</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>4638 rows × 36 columns</p>
</div>



## Appendix

### SIT source data


```python
sit.sit_data.age_classes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>class_size</th>
      <th>start_year</th>
      <th>end_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGEID0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AGEID1</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AGEID2</td>
      <td>5</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AGEID3</td>
      <td>5</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGEID4</td>
      <td>5</td>
      <td>16</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>AGEID196</td>
      <td>5</td>
      <td>976</td>
      <td>980</td>
    </tr>
    <tr>
      <th>197</th>
      <td>AGEID197</td>
      <td>5</td>
      <td>981</td>
      <td>985</td>
    </tr>
    <tr>
      <th>198</th>
      <td>AGEID198</td>
      <td>5</td>
      <td>986</td>
      <td>990</td>
    </tr>
    <tr>
      <th>199</th>
      <td>AGEID199</td>
      <td>5</td>
      <td>991</td>
      <td>995</td>
    </tr>
    <tr>
      <th>200</th>
      <td>AGEID200</td>
      <td>5</td>
      <td>996</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 4 columns</p>
</div>




```python
sit.sit_data.inventory
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Track</th>
      <th>State</th>
      <th>au</th>
      <th>LeadSpecies</th>
      <th>age</th>
      <th>area</th>
      <th>delay</th>
      <th>land_class</th>
      <th>historical_disturbance_type</th>
      <th>last_pass_disturbance_type</th>
      <th>spatial_reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>managed</td>
      <td>2121</td>
      <td>softwood</td>
      <td>69</td>
      <td>2.045750</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>6</td>
      <td>7.437000</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>6</td>
      <td>0.118866</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
      <td>7</td>
      <td>7.194100</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>managed</td>
      <td>2113</td>
      <td>softwood</td>
      <td>8</td>
      <td>4.298460</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1415</th>
      <td>108</td>
      <td>managed</td>
      <td>520</td>
      <td>softwood</td>
      <td>102</td>
      <td>6.028800</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>1865</td>
    </tr>
    <tr>
      <th>1416</th>
      <td>7</td>
      <td>managed</td>
      <td>2122</td>
      <td>softwood</td>
      <td>49</td>
      <td>5.760800</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>1417</th>
      <td>3</td>
      <td>managed</td>
      <td>2104</td>
      <td>softwood</td>
      <td>69</td>
      <td>2.157650</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>1867</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>430</td>
      <td>managed</td>
      <td>451</td>
      <td>softwood</td>
      <td>114</td>
      <td>6.472200</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>1868</td>
    </tr>
    <tr>
      <th>1419</th>
      <td>136</td>
      <td>managed</td>
      <td>2102</td>
      <td>softwood</td>
      <td>34</td>
      <td>2.636840</td>
      <td>0</td>
      <td>0</td>
      <td>Fire</td>
      <td>CC</td>
      <td>1869</td>
    </tr>
  </tbody>
</table>
<p>1420 rows × 11 columns</p>
</div>




```python
sit.sit_data.classifiers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Track</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>State</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>au</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>LeadSpecies</td>
    </tr>
  </tbody>
</table>
</div>




```python
sit.sit_data.classifier_values
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classifier_id</th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>859</th>
      <td>3</td>
      <td>2122</td>
      <td>2122</td>
    </tr>
    <tr>
      <th>860</th>
      <td>3</td>
      <td>2123</td>
      <td>2123</td>
    </tr>
    <tr>
      <th>861</th>
      <td>3</td>
      <td>2124</td>
      <td>2124</td>
    </tr>
    <tr>
      <th>862</th>
      <td>3</td>
      <td>2204</td>
      <td>2204</td>
    </tr>
    <tr>
      <th>863</th>
      <td>4</td>
      <td>softwood</td>
      <td>softwood</td>
    </tr>
  </tbody>
</table>
<p>860 rows × 3 columns</p>
</div>




```python
sit.sit_data.disturbance_types
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default_disturbance_type_id</th>
      <th>sit_disturbance_type_id</th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>204</td>
      <td>1</td>
      <td>CC</td>
      <td>CC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>204</td>
      <td>2</td>
      <td>CC$1</td>
      <td>CC$1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Fire</td>
      <td>Fire</td>
    </tr>
  </tbody>
</table>
</div>




```python
sit.sit_data.yield_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Track</th>
      <th>State</th>
      <th>au</th>
      <th>LeadSpecies</th>
      <th>leading_species</th>
      <th>v0</th>
      <th>v1</th>
      <th>v2</th>
      <th>v3</th>
      <th>v4</th>
      <th>...</th>
      <th>v191</th>
      <th>v192</th>
      <th>v193</th>
      <th>v194</th>
      <th>v195</th>
      <th>v196</th>
      <th>v197</th>
      <th>v198</th>
      <th>v199</th>
      <th>v200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>54.5000</td>
      <td>102.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>7.0</td>
      <td>53.5000</td>
      <td>100.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.5000</td>
      <td>19.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>425</th>
      <td>426</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.8722</td>
      <td>13.7444</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>427</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.2103</td>
      <td>28.4206</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>427</th>
      <td>428</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.1038</td>
      <td>42.2076</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>428</th>
      <td>429</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>430</td>
      <td>?</td>
      <td>?</td>
      <td>softwood</td>
      <td>194</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0669</td>
      <td>4.1338</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>430 rows × 206 columns</p>
</div>




```python
print(json.dumps(sit.config, indent=4, sort_keys=True))
```

    {
        "import_config": {
            "age_classes": {
                "params": {
                    "path": "AgeClasses.csv"
                },
                "type": "csv"
            },
            "classifiers": {
                "params": {
                    "path": "Classifiers.csv"
                },
                "type": "csv"
            },
            "disturbance_types": {
                "params": {
                    "path": "DistType.csv"
                },
                "type": "csv"
            },
            "events": {
                "params": {
                    "path": "DistEvents.csv"
                },
                "type": "csv"
            },
            "inventory": {
                "params": {
                    "path": "Inventory.csv"
                },
                "type": "csv"
            },
            "transitions": {
                "params": {
                    "path": "Transitions.csv"
                },
                "type": "csv"
            },
            "yield": {
                "params": {
                    "path": "Growth.csv"
                },
                "type": "csv"
            }
        },
        "mapping_config": {
            "disturbance_types": {
                "disturbance_type_mapping": [
                    {
                        "default_dist_type": "Clearcut harvesting without salvage",
                        "user_dist_type": "CC"
                    },
                    {
                        "default_dist_type": "Clearcut harvesting without salvage",
                        "user_dist_type": "CC$1"
                    },
                    {
                        "default_dist_type": "Wildfire",
                        "user_dist_type": "Fire"
                    }
                ]
            },
            "nonforest": null,
            "spatial_units": {
                "admin_boundary": "British Columbia",
                "eco_boundary": "Pacific Maritime",
                "mapping_mode": "SingleDefaultSpatialUnit"
            },
            "species": {
                "species_classifier": "LeadSpecies",
                "species_mapping": [
                    {
                        "default_species": "Softwood forest type",
                        "user_species": "softwood"
                    }
                ]
            }
        }
    }



```python

```
