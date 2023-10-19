---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from IPython.display import display, Markdown
import os
import json
from libcbm import resources
import pandas as pd
import numpy as np
from numpy.random import default_rng
from libcbm.model.cbm_exn import cbm_exn_model
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory
from libcbm.model.model_definition.model_variables import ModelVariables
```

```python
rng = default_rng()
n_stands = 10
```

```python
# load the default bundled parameters
parameters = parameters_factory(resources.get_cbm_exn_parameters_dir())
```

# Fully dynamic modelling with cbm_exn

This document illustrates a scheme to simulate with fully dynamic pools, fluxes and flow in libcbm using the cbm_exn package.  

This outlines a component of a highly efficient method for processing C flows in a CBM-Like model, producing outputs that are compatible with current reporting methods, and systems.

The approach makes the hard coded model pool, flow, flux in CBM-CFS3 fully dynamic, and definable via high level language in the cbm_exn package.  

Prior to this improvement, using CBM-CFS3, it was only possible to modify parameters for the pool-flows withing the hard-coded model structure.  This opens up opportunities to evaluate changes to model structure dynamically, in addition to modifiying pool flows.  

In addition the architecture opens opportunities for using machine learning both for informing pool flow parameters and the model structure itself.

The document explains the basic structure of the default parameters and configuration in `cbm_exn`, and also shows the an example of the default pool flows generated in dataframe form, and describes their format. 

While only the default pool flows (based on CBM-CFS3) are presented here, it's important to note if the general format is followed it's possible to alter model pool-flow-flux structure via high level programming language.  The default values could potentially be used as a template to accomplish this.

Sections below are:

 * Operation DataFrame: a description of the storage scheme for mult-dimensional pool flow proportions within dataframe structures.
 * Default spinup operation dataframes: examples of spinup ops derived from the cbm_exn default parameters and example simulation area input
 * Default step operation dataframes: example of step ops also derived from the same inputs
 * Appendix1, Appendix2: the default parameters, and example simulation area used.


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
n_stands = 1000
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
            "mean_annual_temperature":  np.arange(0, 2, 2/n_stands), # make a temperature ramp
            "historical_disturbance_type": np.full(n_stands, 1, dtype="int"),
            "last_pass_disturbance_type": np.full(n_stands, 1, dtype="int"),
        }
    ),
    "increments":increments,
}
```

```python
spinup_vars = cbm_exn_spinup.prepare_spinup_vars(
    ModelVariables.from_pandas(spinup_input),
    parameters,
)
spinup_op_list = cbm_exn_spinup.get_default_op_list()
spinup_ops = cbm_exn_spinup.get_default_ops(parameters, spinup_vars)
```

```python
spinup_op_list
```

# Operation dataframes

Operation dataframes are multi-dimensional sparse matrices, using column-name-formatting to denote the dimensions.  

The rationale for this design apporach is that multi-dimensional arrays lack interoperable standards that would work with many different high level languages simulataneously.  Dataframes however are well supported by several high level languages.

Each dataframe has pool flow columns, where the column name is of the form:

    pool_source_name.pool_sink_name
    
Each row can be interpreted as a pool flow matrix (in sparse coordinate form).  By default unspecified diagnal values (those where `pool_source_name` is equal to `pool_sink_name`) will be assigned a value of 1.

There are 4 basic types of dataframes with regards to mapping to simulation space.  These are covered in the next section

## Single row 

The single row is a matrix that is applied to all simulation areas.  See the `slow_mixing` dataframe in the following section.

## Simulation-aligned 

the dataframe has a row for each simulation area in subsequent spinup or step calls.  Each row represents a matrix that is 1:1 with simulation areas
This type is appropriate for processes that vary by simulation area. See the `dom_decay` dataframe in the followin section.

## Property-indexed 

A property-index dataframe has one or more values that correspond to values stored in the current simulation state. This type of dataframe contains 1 or more columns of the form:

    [table_name.variable_name]

Where a simulation table, series name pair is surrounded by left and right brackets.

An example of this is if one flow matrix is defined for each spatial unit identifier.  See the `disturbance` dataframe in the following section for an example.

## Both Simulation-aligned, and Property-indexed

If a dataframe is both simulation aligned and property indexed each row corresponds to both a simulation area, and 1 or more properties within the simulation areas.  The following pattern of columns is present in this type of dataframe:

    [row_idx], [table_name_1.variable_name_1], ... [table_name_N.variable_name_N]

An example of this is a dataframe of matrices where each row corresponds to the simulation areas, and to simulation age.  See the spinup growth dataframe in the following section.


# Default spinup operation dataframes
The default spinup operation dataframes are generated as a function of the default parameters (see appendix 1) and the simulation area input (Appendix 2).  

The [libcbm.model.cbm_exn.cbm_exn_spinup.spinup](https://github.com/cat-cfs/libcbm_py/blob/700b2febb73681ca2b4456d4db88d5c399008640/libcbm/model/cbm_exn/cbm_exn_spinup.py#L169) function can directly ingest operations in this format via the `ops` parameter.

Within a spinup timestep these operations are applied in the following order by default.  The order and naming of these operation is user-specifyable via passing a string list to the `op_sequence` parameter of the above linked function

 1. growth
 1. snag_turnover
 1. biomass_turnover
 1. overmature_decline
 1. growth
 1. dom_decay
 1. slow_decay
 1. slow_mixing
 1. disturbance

```python
for op in spinup_ops:
    display(Markdown(f"## {op['name']}"))
    display(op["op_data"])
```

Run the spinup routine with the default operations passed as a parameter


```python
with cbm_exn_model.initialize() as model:
    cbm_vars = cbm_exn_spinup.spinup(
        model,
        spinup_vars,
        ops=spinup_ops,
        op_sequence=spinup_op_list
    )
```

```python
# initialize parameters for stepping (values for illustration)
cbm_vars["parameters"]["mean_annual_temperature"].assign(1.1)
cbm_vars["parameters"]["merch_inc"].assign(0.1)
cbm_vars["parameters"]["foliage_inc"].assign(0.01)
cbm_vars["parameters"]["other_inc"].assign(0.05)
cbm_vars["parameters"]["disturbance_type"].assign(
    rng.choice(
        [0,1,4], n_stands, p=[0.98, 0.01, 0.01]
    )
)
```

```python
step_ops_sequence = cbm_exn_step.get_default_annual_process_op_sequence()
step_disturbance_ops_sequence = cbm_exn_step.get_default_disturbance_op_sequence()
step_ops = cbm_exn_step.get_default_ops(parameters, cbm_vars)
```

```python
step_disturbance_ops_sequence
```

```python
step_ops_sequence
```

<!-- #region -->
# Step processes

With the exception of growth and overmature decline, the default step processes use an identical process for generation as the spinup processes, and so only growth and overmature decline are shown below.

The difference is that by default in stepping, growth and overmature decline dataframes are both Simulation-aligned.


The [libcbm.model.cbm_exn.cbm_exn_step.step](https://github.com/cat-cfs/libcbm_py/blob/700b2febb73681ca2b4456d4db88d5c399008640/libcbm/model/cbm_exn/cbm_exn_step.py#L190) function can directly ingest operations in this format via the `ops` parameter
<!-- #endregion -->

```python
for op in step_ops:
    name = op['name']
    if name in ["growth", "overmature_decline"]:
        display(Markdown(f"## {name}"))
        display(op["op_data"])
```

run a timestep with the specified parameters

```python
with cbm_exn_model.initialize() as model:
    cbm_vars = cbm_exn_step.step(
        model,
        cbm_vars,
        ops=step_ops,
        step_op_sequence=step_ops_sequence,
        disturbance_op_sequence=step_disturbance_ops_sequence
    )
```

```python
cbm_vars["pools"].to_pandas()
```

```python
cbm_vars["flux"].to_pandas()
```

## Appendix 1: CBM EXN Default parameters

```python

for k, v in parameters._data.items():
    display(Markdown(f"## {k}"))
    if isinstance(v, list):
        display(Markdown(f"```json\n{json.dumps(v, indent=4)}\n```"))
    else:
        display(v)
```

## Appendix 2: simulation parameters used to generate op-dataframes


### Net C increments

These are derived from Tutorial 1 of the CBM-CFS3 Operational-Scale toolbox.  Each simulation area is assigned the same set of increments.

```python
display(increments)
```

### Simulation areas

```python
display(spinup_input["parameters"])
```

```python

```
