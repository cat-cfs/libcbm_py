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
import numpy as np
import pandas as pd
from types import SimpleNamespace
```

```python
from libcbm import resources
from libcbm.model.moss_c import model_context
from libcbm.model.moss_c import model
from libcbm.model.moss_c import pools
```

```python
data_dir = os.path.join(
   resources.get_test_resources_dir(),
    "moss_c_multiple_stands")
```

```python
ctx = model_context.create_from_csv(data_dir)
```

```python
model.spinup(ctx)
```

```python
pools_by_timestep = pd.DataFrame()
flux_by_timestep = pd.DataFrame()
```

```python
pools_0 = ctx.get_pools_df()
pools_0.insert(0, "t", 0)
pools_by_timestep = pools_by_timestep.append(pools_0)

for t in range(0,100):
    if t == 20:
        # disturb everything to demonstrate how this works
        ctx.state.disturbance_type[:] = 1
    else: 
        ctx.state.disturbance_type[:] = 0
    model.step(ctx)
    
    pools_t = ctx.get_pools_df()
    pools_t.insert(0, "t", t)
    pools_by_timestep = pools_by_timestep.append(pools_t)
    
    flux_t = ctx.flux.copy()
    flux_t.insert(0, "t", t)
    flux_by_timestep = flux_by_timestep.append(flux_t)
    
```

```python
pools_by_timestep.groupby("t").sum()[[p.name for p in pools.ECOSYSTEM_POOLS]].plot(figsize=(10,8))
```

```python
flux_by_timestep[["t","NPPFeatherMoss", "NPPSphagnumMoss"]].groupby("t").sum().plot(figsize=(10,8))

```

```python
flux_by_timestep.groupby("t").sum().plot(figsize=(15,10))
```

```python

```
