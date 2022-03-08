---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.output import InMemoryCBMOutput
from libcbm.model.cbm.stand_cbm_factory import StandCBMFactory
```

```python
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

n_steps = 10

inventory = pd.DataFrame(
    columns=[
        "c1", "c2", "admin_boundary", "eco_boundary", "age", "area",
        "delay", "land_class", "afforestation_pre_type",
        "historic_disturbance_type", "last_pass_disturbance_type"],
    data=[
        ["c1_v1", "c2_v1", "British Columbia", "Pacific Maritime", 15, 1.0,
         0, "UNFCCC_FL_R_FL", None, "Wildfire", "Wildfire"],
        ["c1_v1", "c2_v1", "British Columbia", "Pacific Maritime", 0, 1.0,
         0, "UNFCCC_FL_R_FL", None, "Wildfire", "Wildfire"]])

n_stands = len(inventory.index)

# the following spinup parameters can be specified here directly.
# If they are not specified, they will be pulled from the
# cbm_defaults database on a per-Spatial Unit basis
# these values can be mixes of either scalar, or of the same length
# as inventory.shape[0]
spinup_params = cbm_variables.initialize_spinup_parameters(
    n_stands=n_stands,
    return_interval=np.array([150, 160], dtype=np.int32),
    min_rotations=5,
    max_rotations=10,
    mean_annual_temp=10)


csets, inv = cbm_factory.prepare_inventory(inventory)

with cbm_factory.initialize_cbm() as cbm:

    # can specify a function to gather results from spinup
    # this will incur a performance penalty
    spinup_results = InMemoryCBMOutput(density=True)
    cbm_results = InMemoryCBMOutput(
        classifier_map=cbm_factory.get_classifier_map(),
        disturbance_type_map={1: "fires"})
    cbm_simulator.simulate(
        cbm,
        n_steps=n_steps,
        classifiers=csets,
        inventory=inv,
        pre_dynamics_func=lambda t, cbm_vars: cbm_vars,
        reporting_func=cbm_results.append_simulation_result,
        spinup_params=spinup_params,
        spinup_reporting_func=spinup_results.append_simulation_result)

```

```python
# the spinup results are in a similar format as CBM results
spinup_results.__dict__.keys()
```

```python
spinup_results.pools[["identifier", "timestep", "BelowGroundSlowSoil"]].pivot(index="timestep", columns="identifier").plot()
```

```python
spinup_results.flux[["identifier", "timestep", "DisturbanceCO2Production"]].pivot(index="timestep", columns="identifier").plot()
```

```python
# examine the state variables for stand identifer 1
spinup_results.state[spinup_results.state.identifier==1].set_index("timestep").head()
```
