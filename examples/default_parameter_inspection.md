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

```python
import pandas as pd
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
```

```python
db_path = r"C:\Users\smorken\dev\cbm_defaults\cbm_defaults_2019.db"
cbm_default_parameters = cbm_defaults.load_cbm_parameters(db_path)
cbm_default_ref = CBMDefaultsReference(db_path, "en-CA")

```

```python
spatial_units = pd.DataFrame([dict(x) for x in cbm_default_ref.get_spatial_units()])
pools = pd.DataFrame([dict(x) for x in cbm_default_ref.pools_ref])

```

```python
def load_default_parameters(name):
    json_parameters = cbm_default_parameters[name]
    return pd.DataFrame(data=json_parameters["data"], columns=json_parameters["column_map"])
```

```python
default_parameters ={k: load_default_parameters(k) for k in [
            "decay_parameters",
            "slow_mixing_rate",
            "mean_annual_temp",
            "turnover_parameters",
            "disturbance_matrix_values",
            "disturbance_matrix_associations",
            "root_parameter",
            "growth_multipliers",
            "land_classes",
            "land_class_transitions",
            "spatial_units",
            "random_return_interval",
            "spinup_parameter",
            "afforestation_pre_type"
            ]}
```

```python
pools.merge(default_parameters["decay_parameters"], left_on="id", right_on="Pool")
```

```python
default_parameters["slow_mixing_rate"]
```

```python
spatial_units.merge(default_parameters["mean_annual_temp"])
```

```python
default_parameters["turnover_parameters"]
```

```python
default_parameters["root_parameter"]
```

```python
spatial_units.merge(default_parameters["spinup_parameter"])
```

```python

```
