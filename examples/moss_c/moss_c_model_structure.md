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
import os
import pandas as pd
from types import SimpleNamespace
```

```python
from libcbm import resources
from libcbm.model.moss_c import model_context
from libcbm.model.moss_c import model
```

The moss c pools are defined in a python enumeration, and can be fetched as in the following example

```python
# get the moss C pools
from libcbm.model.moss_c.pools import Pool
for pool in Pool:
    print(pool.name)
```

Some moss c model test cases, example datasets are included in the [libcbm test resources dir](https://github.com/cat-cfs/libcbm_py/tree/master/libcbm/resources/test)

```python
test_data_dir = os.path.join(
   resources.get_test_resources_dir(),
    "moss_c_test_case")
```

Assembling the moss c [model context]()

```python
ctx = model_context.create_from_csv(test_data_dir)
```

Run the model spinup routine.  The model context will be altered in-place by the spinup process.

Setting enable_debugging=True will make the spinup method return a detailed timestep-by-timestep account of the spinup process, but will incur a significant processing and memory consumption cost.

```python
spinup_debug = model.spinup(ctx, enable_debugging=True)
```

```python
spinup_debug.spinup_vars
```

```python
spinup_debug.spinup_vars.set_index("t").plot(figsize=(10,10))
```

```python
spinup_debug.model_state
ms = spinup_debug.model_state.copy()
ms.set_index("t").drop(columns=[]).plot(figsize=(15,10))
```

```python
p = spinup_debug.pools.copy()
p["total_slow"] = p[["FeatherMossSlow", "SphagnumMossSlow"]].sum(axis=1)
p.drop(columns=["Input","CO2","CH4","CO"]).set_index("t").plot(figsize=(15,10))
```

```python
spinup_debug.pools
```
