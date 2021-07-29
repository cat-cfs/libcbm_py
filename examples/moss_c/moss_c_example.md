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
%load_ext snakeviz
```

```python
from libcbm import resources
from libcbm.model.moss_c import model_context
from libcbm.model.moss_c import model
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
%%snakeviz
model.spinup(ctx)
```

```python
ctx.get_pools_df()
```

```python
ctx
```

```python
model.step(ctx)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
dynamics = model.annual_process_dynamics(ctx.state, ctx.params)
```

```python
pd.DataFrame({k:v for k, v in dynamics.__dict__.items()})
```

```python
pd.DataFrame({k:v for k, v in ctx.params.__dict__.items()})
```

```python
model.f7(mean_annual_temp=np.array([-10000]), base_decay_rate=np.array([0.18]), q10=np.array([1]), t_ref=np.array([10]) )
```

```python
ctx.params.mean_annual_temp
```

```python
np.exp(0)
```

```python
np.log(10)
```

```python

```
