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
import json, os
import notebook_startup
```

```python
from libcbm.input.sit import sit_reader
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_simulator
```

```python
settings = notebook_startup.load_settings()
dll_path = settings["libcbm_path"]
db_path = settings["cbm_defaults_db_path"]
cbm_defaults_ref = CBMDefaultsReference(db_path)
```

```python
config_path = os.path.abspath("./sit/growth_only/sit_config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
sit_data = sit_reader.read(config["import_config"], os.path.dirname(config_path))
sit_mapping = SITMapping(config["mapping_config"], cbm_defaults_ref)
```

```python
cbm = sit_cbm_factory.initialize_cbm(
    db_path, dll_path, sit_data, sit_mapping)
```

```python
classifiers, inventory = sit_cbm_factory.initialize_inventory(
    sit_data, sit_mapping)
```

```python
cbm_simulator.simulate(
    cbm, 1, classifiers, inventory, cbm_defaults_ref.get_pools(), cbm_defaults_ref.get_flux_indicators(),
    pre_dynamics_func = lambda x: x, reporting_func=lambda t,x: None)
```

```python

```
