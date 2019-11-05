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

# LibCBM versus CBM-CFS3 growth testing #
This notebook is a automated test of the growth curve implementation in CBM-CFS3 versus that in LibCBM.  The objective is to ensure that LibCBM results match the established CBM-CFS3 model very closely. 

The script automatically generates randomized merchantable volume growth curves in various configurations
 * a random number of merchantable volume curve components, which can result in purely softwood, purely hardwood or mixed stands
 * random spatial unit (which is associated with biomass conversion parameters)
 * one of several random age/volume curve generators with varying amplitudes and start points (impulse, ramp, step, and exp curve)
 
It then compares the results and sorts them by largest different for plotting.

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

```python
from libcbm.test.cbm import case_generation
from libcbm.test.cbm.cbm3_support import cbm3_simulator
from libcbm.test.cbm import test_case_simulator
from libcbm.test.cbm import pool_comparison
from libcbm.test.cbm import result_comparison
```


variables and paths needed to run the tests

```python
settings = notebook_startup.load_settings()
cbm3_exe_path = settings["cbm3_exe_path"]
toolbox_path = settings["toolbox_path"]
archive_index_db_path = settings["archive_index_db_path"]

cbm_defaults_db_path = settings["cbm_defaults_db_path"]
libcbm_path = settings["libcbm_path"]
```

```python
age_interval=5
num_age_classes = 40 #required by cbm3
n_steps = 250
```

generate randomized growth curve test cases

```python
cases = case_generation.generate_scenarios(
    random_seed = 4,
    num_cases = 10,
    db_path = cbm_defaults_db_path,
    n_steps=n_steps,
    max_disturbances = 0,
    max_components = 3,
    n_growth_digits = 2,
    age_interval=age_interval,
    growth_curve_len=age_interval*num_age_classes,
    growth_only=True)

```

run the test cases on libCBM

```python
libcbm_result = test_case_simulator.run_test_cases(
    dll_path=libcbm_path, db_path=cbm_defaults_db_path, cases=cases, n_steps=n_steps)
```

run test cases on cbm-cfs3. uses [StandardImportToolPlugin](https://github.com/cat-cfs/StandardImportToolPlugin) and [cbm3-python](https://github.com/cat-cfs/cbm3_python) to automate cbm-cfs3 functionality

```python
project_path = cbm3_simulator.import_cbm3_project(
    name="growth_curve_testing",
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

join the results for plotting

```python
pools_merged = pool_comparison.get_merged_pools(cbm3_result["pools"], libcbm_result["pools"], "biomass")
```

plot the worst 20 differences

```python
result_comparison.get_summarized_diff_plot(
    merged=pools_merged, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed pool differences [tonnes C/ha]",
    title="Libcbm versus CBM3 Biomass: sum of libcbm minus CBM3 for all timesteps")

```

plot a few of the worst cases for debugging

```python
result_comparison.get_test_case_comparison_plot(
    identifier=2, merged=pools_merged, diff=False,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))
result_comparison.get_test_case_comparison_plot(
    identifier=2, merged=pools_merged, diff=True,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))
```

```python

```

```python

```

```python

```
