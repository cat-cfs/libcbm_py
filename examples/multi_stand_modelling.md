---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
from libcbm.storage import dataframe
from libcbm.model.cbm import stand_cbm_factory
from libcbm.model.cbm import cbm_output
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm import cbm_defaults
from libcbm import resources
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
```
Define merchantable volume and classifiers input used to drive CBM.

The values shown here are purely for illustrative purposes.

Inventory and yields have a m:1 relationship, and the relationship is defined by
the classifier values attached to each merch volume curve and each inventory
record.

This information is well-suited for storage in a `json` formatted file in order
to scale up this information.

```python

classifiers = dict(
    c1=["c1_v1", "c1_v2"],
    c2=["c2_v1"]
)


merch_volumes = [
    dict(
        classifier_set=["c1_v1", "?"],
        merch_volumes=[
          dict(
            species="Spruce",
            age_volume_pairs=[
                [0, 0],
                [50, 100],
                [100, 150],
                [150, 200]
            ]
          ),
          dict(
            species="Oak",
            age_volume_pairs=[
                [0, 0],
                [50, 100],
                [100, 150],
                [150, 200],
            ]
          )
        ]
    ),
    dict(
        classifier_set=["c1_v2", "c2_v1"],
        merch_volumes=[
          dict(
            species="Spruce",
            age_volume_pairs=[
                [0, 0],
                [50, 100],
                [100, 150],
                [150, 200],
            ]
          ),
          dict(
            species="Oak",
            age_volume_pairs=[
                [0, 0],
                [50, 100],
                [100, 150],
                [150, 200],
            ]
          )
        ]
    )
]

```


Define example inventory, this could also be derived from CSV.

```python
n_steps=50
n_stands=1000
rng = np.random.default_rng()
inventory = dataframe.from_pandas(
  pd.DataFrame(dict(
    c1="c1_v1",
    c2="c2_v1",
    admin_boundary="Ontario",
    eco_boundary="Mixedwood Plains",
    age=rng.integers(low=0, high=60, size=n_stands),
    area=1.0,
    delay=0,
    land_class="UNFCCC_FL_R_FL",
    afforestation_pre_type="None",
    historic_disturbance_type="Wildfire",
    last_pass_disturbance_type="Wildfire"
  ))
)
```

Default parameters and defaults reference: inspection and modification of CBM
parameters

```python
defaults_ref = CBMDefaultsReference(resources.get_cbm_defaults_path())
```

Example showing how to interact with one of the properties of `defaults_ref`

```python
defaults_ref.as_data_frame(defaults_ref.disturbance_type_ref)
```

```python
default_parameters = cbm_defaults.get_cbm_parameters_factory(
  resources.get_cbm_defaults_path())()
default_parameters
```
The parameters can be changed here, and will set the value during simulation via
a parameter factory function

```python
# replace the slow mixing rate parameter with a new value
default_parameters["slow_mixing_rate"].iloc[0,0] = 0.005

# create a new "cbm_parameters_factory" with the customizations
def parameter_factory():
    return default_parameters

```

CBM Spinup function - initialize the CBM pools and state (cbm_vars)

```python
def spinup(cbm_factory, parameter_factory, inventory):
    with cbm_factory.initialize_cbm(
        # note we pass the customized parameter factory here
        cbm_parameters_factory=parameter_factory
    ) as cbm:

        csets, inv = cbm_factory.prepare_inventory(inventory)

        cbm_vars = cbm_variables.initialize_simulation_variables(
          csets,
          inv,
          cbm.pool_codes,
          cbm.flux_indicator_codes,
          inv.backend_type
        )

        spinup_vars = cbm_variables.initialize_spinup_variables(
          cbm_vars,
          inv.backend_type,
          spinup_params=None,
          include_flux=False
        )
        cbm.spinup(spinup_vars, reporting_func = None)
        cbm_vars = cbm.init(cbm_vars)

        return cbm_vars

```

CBM Step function - step the annual process and disturbance routine by 1
timestep

```python
def step(cbm_factory, parameter_factory, cbm_vars):
    with cbm_factory.initialize_cbm(
        # note we pass the customized parameter factory here
        cbm_parameters_factory=parameter_factory
    ) as cbm:

        cbm_vars = cbm.step_start(cbm_vars)
        cbm_vars = cbm.step_disturbance(cbm_vars)
        cbm_vars = cbm.step_annual_process(cbm_vars)
        cbm_vars = cbm.step_end(cbm_vars)
        return (cbm_vars)

```


Create a `StandCBMFactory` object for initializing the cbm model, and a
`CBMOutput`object to store the time step output.


```python
cbm_factory = stand_cbm_factory.StandCBMFactory(
    classifiers, merch_volumes
)
output = cbm_output.CBMOutput(
  classifier_map=cbm_factory.classifier_value_names,
  disturbance_type_map=cbm_factory.disturbance_types
)
```

CBM spinup

```python
cbm_vars = spinup(cbm_factory, parameter_factory, inventory)
output.append_simulation_result(0, cbm_vars)
```

CBM time-stepping

```python

for i in range(1,50):

    # set the disturbance type array with randomly drawn disturbance types
    # note the CBM default disturbance type ids are used

    # this is here to illustrate that it's possible to read and inspect and modify
    # the CBM simulation state between timesteps
    disturbance_types = rng.choice(
      [0, 1, 2],
      n_stands,
      replace=True,
      p=[0.95, 0.03, 0.02]
    )

    cbm_vars.parameters["disturbance_type"].assign(disturbance_types)

    step(cbm_factory, parameter_factory, cbm_vars)
    # record the end of timestep result
    output.append_simulation_result(i, cbm_vars)

```

extract the results data frames

```python
pools_output = output.pools.to_pandas()
flux_output = output.flux.to_pandas()
state_output = output.state.to_pandas()
parameters_output = output.parameters.to_pandas()
area = output.area.to_pandas()
```

Plot some output

```python
mean_age = state_output[["timestep", "age"]].groupby("timestep").mean()

mean_age.plot()
```
