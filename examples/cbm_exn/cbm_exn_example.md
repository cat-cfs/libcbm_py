---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
from libcbm import resources
import pandas as pd
import numpy as np
from numpy.random import default_rng
from libcbm.model.cbm_exn import cbm_exn_model
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
```

```python
# define some run methods for use later in the notebook
def spinup(spinup_input):
    with cbm_exn_model.initialize(
        config_path=None, # when None, packaged default parameters are used
        pandas_interface=True,
        include_spinup_debug=False,
    ) as model:
        cbm_vars = model.spinup(spinup_input)
        return cbm_vars

def step(cbm_vars):
    with cbm_exn_model.initialize(
        config_path=None, # when None, packaged default parameters are used
        pandas_interface=True,
        include_spinup_debug=False,
    ) as model:
        cbm_vars = model.step(cbm_vars)
        return cbm_vars
```

```python
rng = default_rng()
```

```python
# read some packaged net increments, derived from a 
# simulation of the same growth curve used in CBM-CFS3 
# tutorial 1
net_increments = pd.read_csv(
    os.path.join(
        resources.get_test_resources_dir(),
        "cbm_exn_net_increments",
        "net_increments.csv"
    )
)
```

```python
net_increments.set_index("age").plot()
```

```python
# the same set of increments are repeated for each stand in this example
n_stands = 500
increments = None
for s in range(n_stands):
    s_increments = net_increments.copy()
    s_increments.insert(0, "row_idx", s)
    s_increments = s_increments.rename(
        columns={
            "SoftwoodMerch": "merch_inc",
            "SoftwoodFoliage": "foliage_inc",
            "SoftwoodOther": "other_inc"
    })
    increments = pd.concat([increments, s_increments])
```

```python
# create the require inputs for spinup
spinup_input = {
    "parameters": pd.DataFrame(
        {
            # random age
            "age": rng.integers(low=0, high=60, size=n_stands, dtype="int"),
            "area": np.full(n_stands, 1, dtype="int"),
            "delay": np.full(n_stands, 0, dtype="int"),
            "return_interval": np.full(n_stands, 125, dtype="int"),
            "min_rotations": np.full(n_stands, 10, dtype="int"),
            "max_rotations": np.full(n_stands, 30, dtype="int"),
            "spatial_unit_id": np.full(n_stands, 17, dtype="int"), # ontario/mixedwood plains
            "species": np.full(n_stands, 20, dtype="int"), # red pine
            "mean_annual_temperature":  np.full(n_stands, 2.55, dtype="float"),
            "historical_disturbance_type": np.full(n_stands, 1, dtype="int"),
            "last_pass_disturbance_type": np.full(n_stands, 1, dtype="int"),
        }
    ),
    "increments":increments,
}
```

```python
#run spinup
cbm_vars = spinup(spinup_input)
```

```python
output_processor = ModelOutputProcessor()
for t in range(50):
    n_stands = len(cbm_vars["state"]["age"].index)
    cbm_vars["parameters"]["mean_annual_temperature"] = 2.55
    cbm_vars["parameters"]["disturbance_type"] = rng.choice([0,1,4], n_stands, p=[0.98, 0.01, 0.01])
    
    increments = net_increments.merge(
        cbm_vars["state"]["age"],
        left_on="age",
        right_on="age",
        how="right"
    ).fillna(0)
    cbm_vars["parameters"]["merch_inc"] = increments["SoftwoodMerch"]
    cbm_vars["parameters"]["foliage_inc"] = increments["SoftwoodFoliage"]
    cbm_vars["parameters"]["other_inc"] = increments["SoftwoodOther"]
    cbm_vars = step(cbm_vars)
    output_processor.append_results(t, ModelVariables.from_pandas(cbm_vars))
```

```python
results = output_processor.get_results()
```

```python
results["pools"].to_pandas()[
    ["timestep", "Merch", "Foliage", "Other", "CoarseRoots", "FineRoots"]
].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
results["pools"].to_pandas()[[
    "timestep",
    'AboveGroundVeryFastSoil',
    'BelowGroundVeryFastSoil',
    'AboveGroundFastSoil',
    'BelowGroundFastSoil',
    'MediumSoil',
    'AboveGroundSlowSoil',
    'BelowGroundSlowSoil',
    'StemSnag',
    'BranchSnag'
]].groupby("timestep").sum().plot(figsize=(10,10))
```

```python
results["flux"].to_pandas()[[
    "timestep",'DisturbanceCO2Production',
    'DisturbanceCH4Production', 'DisturbanceCOProduction'
]].groupby("timestep").sum().plot(figsize=(15,10))
```

```python

```
