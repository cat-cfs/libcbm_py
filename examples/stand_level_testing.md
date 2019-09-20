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
from libcbm.test.cbm import flux_comparison
from libcbm.test.cbm import state_comparison
from libcbm.test.cbm import result_comparison
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
cases
```

```python
libcbm_result = test_case_simulator.run_test_cases(cbm_defaults_db_path, libcbm_path, cases, n_steps, spinup_debug=False)
```

```python

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
merged_state = state_comparison.get_merged_state(cbm3_result["state"], libcbm_result["state"])

```

```python
result_comparison.get_summarized_diff_plot(
    merged=merged_state, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed state variable differences",
    title="Libcbm versus CBM3 Age and landclass: sum of libcbm minus CBM3 for all timesteps")
```

```python
merged_dom = pool_comparison.get_merged_pools(cbm3_result["pools"], libcbm_result["pools"], "dom")
merged_bio = pool_comparison.get_merged_pools(cbm3_result["pools"], libcbm_result["pools"], "biomass")
```

```python
result_comparison.get_summarized_diff_plot(
    merged=merged_bio, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed pool differences [tonnes C/ha]",
    title="Libcbm versus CBM3 Biomass: sum of libcbm minus CBM3 for all timesteps")

result_comparison.get_summarized_diff_plot(
    merged=merged_dom, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed pool differences [tonnes C/ha]",
    title="Libcbm versus CBM3 DOM: sum of libcbm minus CBM3 for all timesteps",
)
```

```python
merged_disturbance_flux = flux_comparison.get_merged_disturbance_flux(cbm3_result["flux"], libcbm_result["flux"])
merged_annual_process_flux = flux_comparison.get_merged_annual_process_flux(cbm3_result["flux"], libcbm_result["flux"])
```

```python
result_comparison.get_summarized_diff_plot(
    merged=merged_annual_process_flux, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed flux differences [tonnes C/ha yr ^ -1]",
    title="Libcbm versus CBM3 Annual process fluxes: sum of libcbm minus CBM3 for all timesteps")

result_comparison.get_summarized_diff_plot(
    merged=merged_disturbance_flux, max_results=20, figsize=(15,10), kind="bar",
    x_label="test case identifer",
    y_label="summed flux differences [tonnes C/ha yr ^ -1]",
    title="Libcbm versus CBM3 disturbance fluxes: sum of libcbm minus CBM3 for all timesteps")
```

```python
test_case_identifier =4
```

```python
cases[test_case_identifier-1]
```

```python
result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_bio, diff=False,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))

result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_bio, diff=True,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))


result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_dom, diff=False,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))

result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_dom, diff=True,
    x_label="time step", y_label="pool value [tonnes C/ha]", figsize=(15,10))
```

```python
result_comparison.get_test_case_comparison_plot(
        identifier=test_case_identifier, merged=merged_annual_process_flux,
        diff=False, x_label="time step", y_label="pool value [tonnes C/ha]",
        figsize=(15,10))

result_comparison.get_test_case_comparison_plot(
        identifier=test_case_identifier, merged=merged_annual_process_flux,
        diff=True, x_label="time step", y_label="pool value [tonnes C/ha]",
        figsize=(15,10))
```

```python
n_disturbance_fluxes = len(merged_disturbance_flux[merged_disturbance_flux["identifier"]==test_case_identifier])
if n_disturbance_fluxes > 0:
    
    result_comparison.get_test_case_comparison_by_indicator_plot(
        identifier=test_case_identifier, merged=merged_disturbance_flux,
        diff=False, timesteps=None, y_label="", kind="bar", figsize=(15,10))

    result_comparison.get_test_case_comparison_by_indicator_plot(
        identifier=test_case_identifier, merged=merged_disturbance_flux,
        diff=True, timesteps=None, y_label="", kind="bar", figsize=(15,10))


```

```python
result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_state, diff=False,
    x_label="time step", y_label="difference", figsize=(15,10))
result_comparison.get_test_case_comparison_plot(
    identifier=test_case_identifier, merged=merged_state, diff=True,
    x_label="time step", y_label="difference", figsize=(15,10))
```

```python
merged_state[merged_state["identifier"]==3]
```

```python

```
