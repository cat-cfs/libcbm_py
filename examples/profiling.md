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
import pandas as pd
import numpy as np
import cProfile
```

```python
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm.model.cbm.stand_cbm_factory import StandCBMFactory
from libcbm.storage import dataframe
```

```python
def run_cbm():
    classifiers = {
        "c1": ["c1_v1"],
        "c2": ["c2_v1"],

    }
    merch_volumes = [
        {
            "classifier_set": ["c1_v1", "?"],
            "merch_volumes": [{
                "species": "Spruce",
                "age_volume_pairs": [
                    [0, 0],
                    [50, 100],
                    [100, 150],
                    [150, 200],
                ]
            }]
        }
    ]

    cbm_factory = StandCBMFactory(classifiers, merch_volumes)

    n_steps = 50
    n_stands = 1000
    inventory = dataframe.from_pandas(pd.DataFrame(
        index=list(range(0, n_stands)),
        columns=[
            "c1", "c2", "admin_boundary", "eco_boundary", "age", "area",
            "delay", "land_class", "afforestation_pre_type",
            "historic_disturbance_type", "last_pass_disturbance_type"],
        data=[
            ["c1_v1", "c2_v1", "Ontario", "Mixedwood Plains", 15, 1.0,
             0, "UNFCCC_FL_R_FL", None, "Wildfire", "Wildfire"]]))

    n_stands = inventory.n_rows

    csets, inv = cbm_factory.prepare_inventory(inventory)

    with cbm_factory.initialize_cbm() as cbm:

        cbm_output = CBMOutput(
            classifier_map=cbm_factory.get_classifier_map(),
            disturbance_type_map={0: "null", 1: "fires"})
        cbm_simulator.simulate(
            cbm,
            n_steps=n_steps,
            classifiers=csets,
            inventory=inv,
            pre_dynamics_func=lambda t, cbm_vars: cbm_vars,
            reporting_func=cbm_output.append_simulation_result,
        )
```

```python
cProfile.run('run_cbm()')
```

```python

```
