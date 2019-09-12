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
import numpy as np
import cProfile 
```

```python
#helpers for integrating this notebook with libcbm
import notebook_startup
```

```python
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
```

```python
settings = notebook_startup.load_settings()
dll_path = settings["libcbm_path"]
db_path = settings["cbm_defaults_db_path"]

```

```python
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
    dll_path=dll_path,
    dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(
        db_path),
    cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(
        db_path),
    merch_volume_to_biomass_factory=merch_volumes,
    classifiers_factory=classifiers)
```

```python
ref = CBMDefaultsReference(db_path)
```

```python
n_stands = 9000

pools = cbm_variables.initialize_pools(n_stands, ref.get_pools())
flux_indicators = cbm_variables.initialize_flux(n_stands, ref.get_flux_indicators())
spinup_params = cbm_variables.initialize_spinup_parameters(n_stands)
spinup_variables = cbm_variables.initialize_spinup_variables(n_stands)
cbm_params = cbm_variables.initialize_cbm_parameters(n_stands)
cbm_state = cbm_variables.initialize_cbm_state_variables(n_stands)
inventory = cbm_variables.initialize_inventory(
    classifiers=pd.DataFrame({
        "c1": np.ones(n_stands, dtype=np.int) * 1
    }),
    inventory=pd.DataFrame({
        "age": np.random.randint(low=0, high=300, size=n_stands, dtype=np.int),
        "spatial_unit": np.ones(n_stands, dtype=np.int) * 42,
        "afforestation_pre_type_id": np.ones(n_stands, dtype=np.int) * -1,
        "land_class": np.ones(n_stands, dtype=np.int) * 0,
        "historical_disturbance_type": np.ones(n_stands, dtype=np.int) * 1,
        "last_pass_disturbance_type": np.ones(n_stands, dtype=np.int) * 1,
        "delay": np.ones(n_stands, dtype=np.int) * 0,
    }))
```

```python
cProfile.run('cbm.spinup(inventory, pools, spinup_variables, spinup_params)')

```

```python
36.280/9*1000/60
```

```python
results = pd.DataFrame(
    columns=["n_stands","time"], 
    data=[(10, 3.150),(50, 3.515),(100, 3.78),(1000, 9.135),
     (2000, 14.921),(4500,29.698),(5000, 33.918),
     (7000,46.470),(9000,58.779)])

```

```python
results.groupby("n_stands").sum().plot()
```

```python
#projected minutes per million stands spinup (single thread)
9.135*1000/60
```

```python
cProfile.run('cbm.init(inventory, pools, cbm_state)')
```

```python
cProfile.run('for i in range(0,200): cbm.step(inventory, pools, flux_indicators, cbm_state, cbm_params)')
```

```python

```
