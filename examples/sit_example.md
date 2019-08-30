---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import notebook_startup
```

```python
import os
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_disturbance_type_parser
from libcbm.input.sit import sit_age_class_parser
from libcbm.input.sit import sit_inventory_parser
from libcbm.input.sit import sit_yield_parser
from libcbm.input.sit import sit_disturbance_event_parser
```

```python
classifiers_table = pd.DataFrame(
    data=[
        ("1","_CLASSIFIER","classifier1",np.nan,np.nan),
        (1,"a","a",np.nan,np.nan),
        (1,"b","b",np.nan,np.nan),
        (1,"agg1","agg2","a","b"),        
        (1,"agg2","agg2","a","b"),        
        (2,"_CLASSIFIER","classifier2",np.nan,np.nan),
        (2,"a","a",np.nan,np.nan),
        (2,"agg1","agg1","a",np.nan)])

classifiers_table
```

```python

classifiers,classifier_values,classifier_aggregates = sit_classifier_parser.parse(classifiers_table)
```

```python
classifiers
```

```python
classifier_values
```

```python
classifier_aggregates
```

```python
disturbance_types_table = pd.DataFrame(
    data=[
        ("dist1","fire"),
        ("dist2", "clearcut"),
        ("dist3", "clearcut")
    ])
```

```python
disturbance_types_table
```

```python
disturbance_types = sit_disturbance_type_parser.parse(disturbance_types_table)
disturbance_types
```

```python
age_class_table = sit_age_class_parser.generate_sit_age_classes(10, 100)
```

```python
age_class_table = pd.DataFrame(data={"a":[f"age_{x}" for x in range(0,10)], "b":[0]+[10]*9})
```

```python
age_class_table
```

```python
age_classes = sit_age_class_parser.parse(age_class_table)
age_classes
```

```python
inventory_table = pd.DataFrame(
    data=[
        ("b","a","True","age_2",1,1,1,"dist1","dist2",-1),
        ("a","a",False,100,1,0,0,"dist2","dist1",0),
        ("a","a","-1",4,1,0,0,"dist1","dist1",-1)])
```

```python
inventory_table
```

```python
land_classes = {0: "land_class_1", 1: "land_class_2"}
```

```python
inventory = sit_inventory_parser.parse(
    inventory_table, classifiers, classifier_values, disturbance_types, age_classes, land_classes)

```

```python

```

```python
yield_table = pd.DataFrame([
    ("a", "?", "sp1", 0, 10, 20, 30)
])
```

```python
sit_yield_parser.parse(yield_table, classifiers, classifier_values)
```

```python
num_eligibility_cols = len(sit_format.get_disturbance_eligibility_columns(0))
event = {"classifier_set": ["a","?"],
         "age_eligibility": ["False", -1,-1,-1,-1],
         "eligibility": [-1]*num_eligibility_cols,
         "target": [1.0, "1", "A", 100, "dist1", 2]
        }
disturbance_event_table = pd.DataFrame([
    event["classifier_set"] + event["age_eligibility"] + event["eligibility"] + event["target"]
])
disturbance_event_table
```

```python
sit_disturbance_event_parser.parse(disturbance_event_table, classifiers, classifier_values, classifier_aggregates, disturbance_types, age_classes)
```

```python

```

```python

```

```python

```
