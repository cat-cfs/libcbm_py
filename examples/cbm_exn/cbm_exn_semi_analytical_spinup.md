---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import json
from libcbm import resources
import pandas as pd
import numpy as np
from numpy.random import default_rng
from libcbm.model.cbm_exn import cbm_exn_model
from libcbm.model.cbm_exn.semianalytical_spinup import semianalytical_spinup
from libcbm.model.cbm_exn.semianalytical_spinup.semianalytical_spinup import InputMode
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
```

```python
def spinup_semianalytical(spinup_input, parameters):
    return semianalytical_spinup.semianalytical_spinup(
        spinup_input, InputMode.MaxDefinedAge, parameters
    )
```

```python
# define some run methods for use later in the notebook
def spinup(spinup_input, parameters):
    with cbm_exn_model.initialize(
        parameters=parameters, # when None, packaged default parameters are used
        include_spinup_debug=False,
    ) as model:
        cbm_vars = model.spinup(spinup_input)
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
# the same set of increments are repeated for each stand in this example
n_stands = 100
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
# create the required inputs for spinup
ri = rng.integers(low=125, high=126, size=n_stands, dtype="int")
spinup_input = {
    "parameters": pd.DataFrame(
        {
            # random age
            "age": 125,
            "area": np.full(n_stands, 1, dtype="int"),
            "delay": np.full(n_stands, 0, dtype="int"),
            "return_interval": ri,
            "min_rotations": np.full(n_stands, 10, dtype="int"),
            "max_rotations": np.full(n_stands, 30, dtype="int"),
            "spatial_unit_id": np.full(n_stands, 17, dtype="int"), # ontario/mixedwood plains
            "species": np.full(n_stands, 20, dtype="int"), # red pine
            "mean_annual_temperature": rng.uniform(-2, 2, n_stands),
            "historical_disturbance_type": np.full(n_stands, 1, dtype="int"),
            "last_pass_disturbance_type": np.full(n_stands, 1, dtype="int"),
        }
    ),
    "increments":increments,
}
```

```python
# this is the path to some default bundled parameters for cbm_exn
param_path = resources.get_cbm_exn_parameters_dir()
parameters = dict(
    pools=json.load(open(os.path.join(param_path, "pools.json"), 'r')),
    flux=json.load(open(os.path.join(param_path, "flux.json"), 'r')),
    slow_mixing_rate=pd.read_csv(os.path.join(param_path, "slow_mixing_rate.csv")),
    turnover_parameters=pd.read_csv(os.path.join(param_path, "turnover_parameters.csv")),
    species=pd.read_csv(os.path.join(param_path, "species.csv")),
    root_parameters=pd.read_csv(os.path.join(param_path, "root_parameters.csv")),
    decay_parameters=pd.read_csv(os.path.join(param_path, "decay_parameters.csv")),
    disturbance_matrix_value=pd.read_csv(os.path.join(param_path, "disturbance_matrix_value.csv")),
    disturbance_matrix_association=pd.read_csv(os.path.join(param_path, "disturbance_matrix_association.csv")),
)
```

```python
#run spinup
cbm_vars = spinup(spinup_input, parameters)
```

```python
semi_analytical_result = spinup_semianalytical(spinup_input, parameters)
dom_pool_columns = list(semi_analytical_result.columns)
```

```python
spinup_seed_pools = pd.DataFrame(
    {
        p: semi_analytical_result[p] if p in dom_pool_columns else 0.0 for p in parameters["pools"] 
    }
)
spinup_seed_pools["Input"] = 1.0
```

```python
seed_spinup_input = spinup_input.copy()
seed_spinup_input["pools"] = spinup_seed_pools
cbm_vars_seeded = spinup(seed_spinup_input, parameters)
```

```python
cbm_vars_seeded["pools"][dom_pool_columns]
```

```python
cbm_vars["pools"][dom_pool_columns]
```

```python
semi_analytical_result
```

```python

merged = cbm_vars["pools"][dom_pool_columns].merge(
    semi_analytical_result, left_index=True, right_index=True, suffixes=("_iterative", "_semi_analytical")
)
```

```python
import matplotlib.pyplot as plt
import math

def plot(x_y_pairs):
    fig, axes = plt.subplots(3,3, figsize=(10,10))
    fig.suptitle("comparison")
    plt.axis('square')
    
    for i, (name, x,y) in enumerate(x_y_pairs):
        ix = i//3
        iy = i%3
        
        lim_min = math.floor(min(x.min(), y.min()))
        lim_max = math.ceil(max(x.max(), y.max()))
        colors = ['k']*len(x)   
        ax = axes[ix,iy]
        ax.scatter(x, y, c=colors, alpha=0.5)
        ax.set_xlim((lim_min,lim_max))
        ax.set_ylim((lim_min,lim_max))
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
        ax.set(title=name)
        #ax.grid(which='major', color='k', linestyle='--')

        
    fig.tight_layout()

x_y_pairs = []
for d in dom_pool_columns:
    x_y_pairs.append((d, merged[f"{d}_semi_analytical"], merged[f"{d}_iterative"]))
plot(x_y_pairs)
    
```

```python

```
