---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm import resources
```

```python
#change this if you have another database path
db_path = resources.get_cbm_defaults_path()

default_parameters = cbm_defaults.load_cbm_parameters(db_path)
cbm_default_ref = CBMDefaultsReference(db_path, "fr-CA")
```

```python
spatial_units = pd.DataFrame([dict(x) for x in cbm_default_ref.get_spatial_units()])
pools = pd.DataFrame([dict(x) for x in cbm_default_ref.pools_ref])
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
