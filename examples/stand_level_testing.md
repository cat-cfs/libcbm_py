---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# LibCBM versus CBM-CFS3 Stand level testing

```python
import os, json, math
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
#helpers for integrating this notebook with libcbm
import notebook_startup
```

libCBM related imports


```python
from libcbm.test.cbm import case_generation
from libcbm.test.cbm.cbm3_support import cbm3_simulator
from libcbm.test.cbm import test_case_simulator
from libcbm.test.cbm import pool_comparison
```

```python
settings = notebook_startup.load_settings()
cbm3_exe_path = settings["cbm3_exe_path"]
toolbox_path = settings["toolbox_path"]
archive_index_db_path = settings["archive_index_db_path"]
cbm_defaults_db_path = settings["cbm_defaults_db_path"]
libcbm_path = settings["libcbm_path"]
```

```python
age_interval = 5
num_age_classes = 40 #required by cbm3
n_steps = 250
```

generate random test cases

```python
cases = case_generation.generate_scenarios(
    random_seed = 1,
    num_cases = 5,
    db_path = cbm_defaults_db_path,
    n_steps = n_steps,
    max_disturbances = 3,
    max_components = 1,
    n_growth_digits = 2,
    age_interval=age_interval,
    growth_curve_len=age_interval * num_age_classes)

```

```python
libcbm_result = test_case_simulator.run_test_cases(cbm_defaults_db_path, libcbm_path, cases, n_steps, spinup_debug=False)
```

```python

project_path = cbm3_simulator.import_cbm3_project(
    name="stand_level_testing",
    cases=cases,
    age_interval=age_interval,
    num_age_classes=num_age_classes,
    n_steps=n_steps,
    toolbox_path=toolbox_path,
    archive_index_db_path=archive_index_db_path)

cbm3_results_path = cbm3_simulator.run_cbm3(
    archive_index_db_path=archive_index_db_path, 
    project_path=project_path,
    toolbox_path=toolbox_path,
    cbm_exe_path=cbm3_exe_path)

cbm3_result = cbm3_simulator.get_cbm3_results(cbm3_results_path)
```

```python
pools_merged, pool_diffs = pool_comparison.join_pools(libcbm_result["pools"], cbm3_result["pools"], "all")
```

```python
pool_diffs_totals = pool_diffs.drop(columns="timestep")
pool_diffs_totals \
    .groupby("identifier").sum() \
    .sort_values("abs_total_diff", ascending=False) \
    .head(20) \
    .plot(figsize=(15,10), kind="bar")
```

```python
def plot_diff(id):
    markers = ["o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d"]
    bio_pools = pools_merged[pools_merged["identifier"]==id]
    bio_pools = bio_pools.drop(columns="identifier")
    bio_pools = bio_pools.groupby("timestep").sum()
    ax = bio_pools.plot(figsize=(15,12), title=case_generation.get_classifier_value_name(id))
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i%len(markers)])
    ax.legend(ax.get_lines(), bio_pools.columns, loc='best')
    bio_diffs = pool_diffs[pool_diffs["identifier"]==id]
    bio_diffs = bio_diffs.drop(columns="identifier")
    bio_diffs.groupby("timestep").sum() \
        .plot(figsize=(15,12), title=case_generation.get_classifier_value_name(id))
```

```python
plot_diff(4)
plot_diff(5)

```

Spinup debug


```python
cases[7]
```

```python


if "spinup_debug" in libcbm_result:
    libCBM_spinup_debug = libcbm_result["spinup_debug"]
    libCBM_spinup_debug[libCBM_spinup_debug["index"]==7].groupby("iteration").sum().plot(figsize=(10,10))
```

```python

```

```python
libcbm_pools = libcbm_result["pools"]
```

```python
libcbm_pools[libcbm_pools["identifier"]=="8"][["timestep","SoftwoodMerch","HardwoodMerch"]].groupby("timestep").sum().plot()
```

```python
cbm3_pools = cbm3_result["pools"]
```

```python
cbm3_pools[cbm3_pools["identifier"]=="8"] \
    [["TimeStep","Softwood Merchantable","Hardwood Merchantable"]] \
    .groupby("TimeStep").sum().plot()
```

```python
libcbm_state_variables = libcbm_result["state_variable_result"]
```

```python
libcbm_state_variables[libcbm_state_variables["identifier"]=='8']
```

```python

```

```python

```
