---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import cProfile 
import plotly.graph_objects as go 
%matplotlib inline
```

```python
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm import resources
```

```python
db_path = resources.get_cbm_defaults_path()
classifiers = lambda : cbm_config.classifier_config([
    cbm_config.classifier(
        "c1",
        values=[cbm_config.classifier_value("c1_v1")])
])

merch_volumes = lambda : cbm_config.merch_volume_to_biomass_config(
    db_path=db_path,
    merch_volume_curves=[
        cbm_config.merch_volume_curve(
        classifier_set=["c1_v1"],
        merch_volumes=[{
            "species_id": 1,
            "age_volume_pairs": [[0,0],[50,100],[100,150],[150,200]]
        }]
    )]
)
```

```python
cbm = cbm_factory.create(
    dll_path=resources.get_libcbm_bin_path(),
    dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(db_path),
    cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(db_path),
    merch_volume_to_biomass_factory=merch_volumes,
    classifiers_factory=classifiers)
```

```python
ref = CBMDefaultsReference(db_path)
```

```python
n_stands = 1000

pools = cbm_variables.initialize_pools(n_stands, ref.get_pools())
flux_indicators = cbm_variables.initialize_flux(n_stands, ref.get_flux_indicators())
spinup_params = cbm_variables.initialize_spinup_parameters(n_stands)
spinup_variables = cbm_variables.initialize_spinup_variables(n_stands)
cbm_params = cbm_variables.initialize_cbm_parameters(n_stands)
cbm_state = cbm_variables.initialize_cbm_state_variables(n_stands)
inventory = cbm_variables.initialize_inventory(
    inventory=pd.DataFrame({
        "age": np.random.randint(low=0, high=300, size=n_stands, dtype=int),
        "area": np.ones(n_stands),
        "spatial_unit": np.ones(n_stands, dtype=int) * 42,
        "afforestation_pre_type_id": np.ones(n_stands, dtype=int) * -1,
        "land_class": np.ones(n_stands, dtype=int) * 0,
        "historical_disturbance_type": np.ones(n_stands, dtype=int) * 1,
        "last_pass_disturbance_type": np.ones(n_stands, dtype=int) * 1,
        "delay": np.ones(n_stands, dtype=int) * 0,
    }))
classifiers = classifiers=pd.DataFrame({
        "c1": np.ones(n_stands, dtype=int) * 1
    })
```

```python
cProfile.run('cbm.spinup(classifiers, inventory, pools, spinup_variables, spinup_params)')
```

```python
cProfile.run('cbm.init(inventory, pools, cbm_state)')
```

```python
cProfile.run('for i in range(0,200): cbm.step(classifiers, inventory, pools, flux_indicators, cbm_state, cbm_params)')
```

# total time used for numeric processing of 1000 stands 
  - spinup: 4.983 seconds
  - initialization and CBM 200 timesteps: 0.774 seconds
  - total: 5.757 seconds

```python
def project_time(time_per_single_thread_stand, n_projected_stands, max_threads):
    time_single_thread = time_per_single_thread_stand * n_projected_stands
    n_threads_axis = list(range(1,max_threads+1))
    time_axis = [time_single_thread/x for x in n_threads_axis]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_threads_axis,y=time_axis        
    ))
    fig.update_layout(yaxis_type="log")
    fig.show()
    return pd.DataFrame({"n_threads": n_threads_axis, "time [s]": time_axis})
```

Projected time to run 1 million stands through spinup and 200 CBM timesteps by number of threads

```python
project_time(5.757/1000, 1e6, 200)
```

```python

```

```python
results = pd.DataFrame(
    columns=["n_stands","time"], 
    data=[(10, 3.150),(50, 3.515),(100, 3.78),(1000, 9.135),
     (2000, 14.921),(4500,29.698),(5000, 33.918),
     (7000,46.470),(9000,58.779)])
results.groupby("n_stands").sum().plot()
```

```python

```
