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
import numpy as np
import pandas as pd
```

```python
from libcbm.model.model_definition import model
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.model_definition.cbm_variables import CBMVariables
```

```python
pool_def = [
    "Input",
    "WoodyBiomass",
    "Foliage",
    "SlowDOM",
    "MediumDOM",
    "FastDOM",
    "CO2",
    "Products"
]
```

```python
processes = {
    "GrowthAndMortality": 0,
    "Decay": 1,
    "Disturbance": 2
}
```

```python
flux_indicators = [
    {
        "name": "NPP",
        "process": processes["GrowthAndMortality"],
        "source_pools": [
            "Input",
            "Foliage"
        ],
        "sink_pools": [
            "WoodyBiomass",
            "Foliage",
            "FastDOM",
        ]
    },
    {
        "name": "DecayEmissions",
        "process": processes["Decay"],
        "source_pools": [
            "SlowDOM",
            "MediumDOM",
            "FastDOM",
        ],
        "sink_pools": ["CO2"]
    },
    {
        "name": "DisturbanceEmissions",
        "process": processes["Disturbance"],
        "source_pools": [
            "WoodyBiomass",
            "Foliage",
            "SlowDOM",
            "MediumDOM",
            "FastDOM"
        ],
        "sink_pools": ["CO2"]
    },
    {
        "name": "HarvestProduction",
        "process": processes["Disturbance"],
        "source_pools": [
            "WoodyBiomass",
            "Foliage",
            "MediumDOM",
        ],
        "sink_pools": ["Products"]
    }
]
```

```python
def weibull_cumulative(x, k=2.3, y=1):
    c = np.power(x/y, k)
    return 1-np.exp(-c)

def get_npp_matrix(model, age):
    # creates NPP flows based on an age passed to the cumulative weibull distribution
    n_stands = age.shape[0]
    npp = weibull_cumulative((age+1)/100.0) - weibull_cumulative(age/100.0)
    op = model.create_operation(
        matrices=[
            ["Input", "WoodyBiomass", npp],
            ["Input", "Foliage", npp/10.0],
        ],
        fmt="repeating_coordinates",
        matrix_index=np.arange(0, n_stands),
        process_id=processes["GrowthAndMortality"]
    )

    return op
```

```python
pd.DataFrame([weibull_cumulative(x,5,1) for x in np.arange(0,2.5,0.01)]).plot()
```

```python

def get_mortality_matrix(model, n_stands):

    op = model.create_operation(
        matrices=[
            ["WoodyBiomass", "WoodyBiomass", 1.0],
            ["WoodyBiomass", "MediumDOM", 0.01],
            ["Foliage", "Foliage", 1.0],
            ["Foliage", "FastDOM", 0.95],
        ],
        fmt="repeating_coordinates",

        # set every stand to point at the 0th matrix:
        # they all share the same simple mortality matrix
        matrix_index=np.full(n_stands, 0),
        process_id=processes["GrowthAndMortality"]
    )

    return op
```

```python
def get_decay_matrix(model, n_stands):
    op = model.create_operation(
        matrices=[
            ["SlowDOM", "SlowDOM", 0.99],
            ["SlowDOM", "CO2", 0.01],

            ["MediumDOM", "MediumDOM", 0.85],
            ["MediumDOM", "SlowDOM", 0.10],
            ["MediumDOM", "CO2", 0.05],

            ["FastDOM", "FastDOM", 0.65],
            ["FastDOM", "MediumDOM", 0.25],
            ["FastDOM", "CO2", 0.10],
        ],
        fmt="repeating_coordinates",
        matrix_index=np.full(n_stands, 0),
        process_id=processes["Decay"]
    )

    return op
```

```python
disturbance_type_ids = {
    "none": 0,
    "fire": 1,
    "harvest": 2
}

def get_disturbance_matrix(model, disturbance_types):

    no_disturbance = [

    ]
    fire_matrix = [
        ["WoodyBiomass", "WoodyBiomass", 0.0],
        ["WoodyBiomass", "CO2", 0.85],
        ["WoodyBiomass", "MediumDOM", 0.15],
        ["Foliage", "Foliage", 0.0],
        ["Foliage", "CO2", 0.95],
        ["Foliage", "FastDOM", 0.05]
    ]
    harvest_matrix = [
        ["WoodyBiomass", "WoodyBiomass", 0.0],
        ["WoodyBiomass", "Products", 0.85],
        ["WoodyBiomass", "MediumDOM", 0.15],
        ["Foliage", "Foliage", 0.0],
        ["Foliage", "FastDOM", 1.0]
    ]

    op = model.create_operation(
        matrices=[
            no_disturbance, fire_matrix, harvest_matrix
        ],
        fmt="matrix_list",
        matrix_index=disturbance_types,
        process_id=processes["Disturbance"]

    )

    return op


```

```python
with model.initialize(pool_def, flux_indicators) as cbm_model:
    rng = np.random.default_rng()
    output_processor = ModelOutputProcessor()
    n_stands = 10
    model_vars = CBMVariables.from_pandas(
        {
            "pools": pd.DataFrame(
                columns=cbm_model.pool_names,
                data={
                    p: np.zeros(n_stands) for p in cbm_model.pool_names
                },
            ),
            "flux": pd.DataFrame(
                columns=cbm_model.flux_names,
                data={
                    f: np.zeros(n_stands) for f in cbm_model.flux_names
                },
            ),
            "state": pd.DataFrame(
                columns=["enabled"], data=np.ones(n_stands, dtype="int")
            ),
        }
    )
    model_vars["pools"]["Input"].assign(1.0)

    stand_age = np.full(n_stands, 0)

    for t in range(0, 1000):

        # add some simplistic disturbance scheduling

        if (t % 150) == 0:
            disturbance_types = np.full(n_stands, disturbance_type_ids["fire"])
        elif t == 950:
            disturbance_types = np.full(n_stands, disturbance_type_ids["harvest"])
        else:
            disturbance_types = np.full(n_stands, disturbance_type_ids["none"])

        # reset flux at start of every time step
        model_vars["flux"].zero()

        # prepare the matrix operations
        operations = [
            get_disturbance_matrix(cbm_model, disturbance_types),
            get_npp_matrix(cbm_model, stand_age),
            get_mortality_matrix(cbm_model, n_stands),
            get_decay_matrix(cbm_model, n_stands),
        ]

        # enabled can be used to disable(0)/enable(1) dynamics per index
        model_vars["state"]["enabled"].assign(np.full(n_stands, 1))

        cbm_model.compute(model_vars, operations)
        for op in operations:
            op.dispose()
        output_processor.append_results(t, model_vars)
        stand_age[disturbance_types != 0] = 0
        stand_age += 1


```

```python
results = output_processor.get_results()
```

```python
pools = results["pools"]
flux = results["flux"]
```

```python
pools.to_pandas()[
    ['timestep','WoodyBiomass', 'Foliage', 'SlowDOM',
     'MediumDOM', 'FastDOM']
].groupby("timestep").sum().plot(figsize=(15,10))
```

```python
flux.to_pandas()[
    ['timestep', 'NPP', 'DecayEmissions', 'DisturbanceEmissions',
     'HarvestProduction']
].groupby("timestep").sum().plot(figsize=(15,10))
```

```python

```

```python

```

```python

```
