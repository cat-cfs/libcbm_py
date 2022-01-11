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
import numpy as np
import pandas as pd
```

```python
from libcbm.model import model_definition
```

```python
pool_def = {
    "Input": 0,
    "WoodyBiomass": 1,
    "Foliage": 2,
    "SlowDOM": 3,
    "MediumDOM": 4,
    "FastDOM": 5,
    "CO2": 6,
    "Products": 7
}
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
        "process_id": processes["GrowthAndMortality"],
        "source_pools": [
            pool_def["Input"],
            pool_def["Foliage"]
        ],
        "sink_pools": [
            pool_def["WoodyBiomass"],
            pool_def["Foliage"],
            pool_def["FastDOM"],
        ]   
    },
    {
        "name": "DecayEmissions",
        "process_id": processes["Decay"],
        "source_pools": [
            pool_def["SlowDOM"],
            pool_def["MediumDOM"],
            pool_def["FastDOM"],
        ],
        "sink_pools": [pool_def["CO2"]]
    },
    {
        "name": "DisturbanceEmissions",
        "process_id": processes["Disturbance"],
        "source_pools": [
            pool_def["WoodyBiomass"],
            pool_def["Foliage"],
            pool_def["SlowDOM"],
            pool_def["MediumDOM"],
            pool_def["FastDOM"]
        ],
        "sink_pools": [pool_def["CO2"]]
    },
    {
        "name": "HarvestProduction",
        "process_id": processes["Disturbance"],
        "source_pools": [
            pool_def["WoodyBiomass"],
            pool_def["Foliage"],
            pool_def["MediumDOM"],
        ],
        "sink_pools": [pool_def["Products"]]
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
            ["Input", "WoodyBiomass", np.full(n_stands, npp)],
            ["Input", "Foliage", np.full(n_stands, npp/10.0)],
        ], 
        fmt="repeating_coordinates")
    
    op.set_matrix_index(np.arange(0, n_stands))
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
        fmt="repeating_coordinates"
    )
    # set every stand to point at the 0th matrix: 
    # they all share the same simple mortality matrix
    op.set_matrix_index(np.full(n_stands, 0))
    return op
```

```python
def get_decay_matrix(model, n_stands):
    op = model.create_operation(
        matrices=[
            ["SlowDOM", "SlowDOM", 0.97],
            ["SlowDOM", "CO2", 0.03],
            
            ["MediumDOM", "MediumDOM", 0.85],
            ["MediumDOM", "SlowDOM", 0.10],
            ["MediumDOM", "CO2", 0.05],
            
            ["FastDOM", "FastDOM", 0.65],
            ["FastDOM", "MediumDOM", 0.25],
            ["FastDOM", "CO2", 0.10],
        ],
        fmt="repeating_coordinates"
    )
    op.set_matrix_index(np.full(n_stands, 0))
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
        fmt="matrix_list"
    )
    op.set_matrix_index(disturbance_types)
    return op
    
    
```

```python
with model_definition.create_model(pool_def, flux_indicators) as model:

    output_processor = model.create_output_processor()
    n_stands = 100
    pools = model.allocate_pools(n_stands)
    pools[:, pool_def["Input"]] = 1.0

    flux = model.allocate_flux(n_stands)
    stand_age = np.full(n_stands, 0)
    
    for t in range(0, 100):
        
        # add some simplistic disturbance scheduling
        if t == 85:
            disturbance_types = np.full(n_stands, disturbance_type_ids["fire"])  
        elif t == 50:
            disturbance_types = np.full(n_stands, disturbance_type_ids["harvest"])
        else:
            disturbance_types = np.full(n_stands, disturbance_type_ids["none"])
        
        # reset flux at start of every time step
        flux[:] = 0.0
        
        # prepare the matrix operations
        operations = [
            get_disturbance_matrix(model, disturbance_types),
            get_npp_matrix(model, stand_age),
            get_mortality_matrix(model, n_stands),
            get_decay_matrix(model, n_stands),            
        ]

        # associate each above operations with a flux indicator category
        op_processes = [
            processes["Disturbance"],
            processes["GrowthAndMortality"],
            processes["GrowthAndMortality"],
            processes["Decay"],            
        ]

        # enabled can be used to disable(0)/enable(1) dynamics per index
        enabled=np.full(n_stands, 1)
        
        model.compute(pools, flux, operations, op_processes, enabled)
        for op in operations:
            op.dispose()
        output_processor.append_results(t, pools, flux)
        stand_age[disturbance_types != 0] = 0
        stand_age += 1
        

```

```python
output_processor.pools.groupby("timestep").sum().plot(figsize=(15,10))
```

```python
output_processor.flux.groupby("timestep").sum().plot(figsize=(15,10))
```

```python

```
