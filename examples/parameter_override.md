---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os, json
import pandas as pd
from numpy import random
import numpy as np
%matplotlib inline
```

```python
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm import cbm_defaults
from libcbm import resources
```

```python
rng = random.default_rng()
```

```python

def run_cbm_random():
    # runs CBM tutorial2 with randomly drawn stem anual turnover rate, to illustrate how to do this
    config_path = os.path.join(resources.get_test_resources_dir(), "cbm3_tutorial2", "sit_config.json")
    sit = sit_cbm_factory.load_sit(config_path)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)

    # get the default CBM parameters
    default_parameters = sit.defaults.get_parameters_factory()()

    def random_parameter_factory():
        # update the default parameters table "turnover_parameters"
        
        #first convert the json formatted table to a dataframe
        turnover_parameters = cbm_defaults.parameter_as_dataframe(default_parameters["turnover_parameters"])
        #fetch the StemAnnualTurnoverRate series from the dataframe
        stem_turnover_rate = turnover_parameters["StemAnnualTurnoverRate"]
        # replace the StemAnnualTurnoverRate with values drawn from a distribution
        turnover_parameters["StemAnnualTurnoverRate"] = rng.normal(stem_turnover_rate, stem_turnover_rate / 20.0)
        # convert the dataframe back to the json format used by CBM
        default_parameters["turnover_parameters"] = cbm_defaults.dataframe_as_parameter(turnover_parameters)
        return default_parameters

    # use the above random parameter factory as the source for CBM parameters
    cbm = sit_cbm_factory.initialize_cbm(sit, parameters_factory=random_parameter_factory)
    
    results, reporting_func = cbm_simulator.create_in_memory_reporting_func()
    rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)

    cbm_simulator.simulate(
        cbm,
        n_steps              = 20,
        classifiers          = classifiers,
        inventory            = inventory,
        pool_codes           = sit.defaults.get_pools(),
        flux_indicator_codes = sit.defaults.get_flux_indicators(),
        pre_dynamics_func    = rule_based_processor.pre_dynamic_func,
        reporting_func       = reporting_func
    )
    # return a single pool value to illustrate the effect of the randomly drawn parameter
    return results.pools[["timestep", "SoftwoodStemSnag"]].groupby("timestep").sum()
```

```python
# run 20 iterations of the random parameter override 
results = {f"sw_stem_snag_iteration_{i}": run_cbm_random()["SoftwoodStemSnag"] for i in range(1,20+1)}
```

```python
pd.DataFrame(results).plot(figsize=(15,10))
```

```python

```
