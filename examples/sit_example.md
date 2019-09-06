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
import notebook_startup
```

```python
import os
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
from libcbm.input.sit import sit_format,sit_parser,sit_classifier_parser, \
    sit_disturbance_type_parser,sit_age_class_parser,sit_inventory_parser, \
    sit_yield_parser,sit_disturbance_event_parser,sit_transition_rule_parser
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference



```

```python
settings = notebook_startup.load_settings()
dll_path = settings["libcbm_path"]
db_path = settings["cbm_defaults_db_path"]
cbm_defaults_ref = CBMDefaultsReference(db_path)
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
        ("b","a","True","age_2",1,0,0,"dist1","dist2",-1),
        ("a","a",False,100,1,0,0,"dist2","dist1",0),
        ("a","a","-1",0,1,10,1,"dist1","dist1",-1)])
```

```python
inventory_table
```

```python
land_classes = {0: "UNFCCC_FL_R_FL", 1: "UNFCCC_CL_R_CL"}
```

```python
inventory = sit_inventory_parser.parse(
    inventory_table, classifiers, classifier_values, disturbance_types, age_classes, land_classes)
inventory
```

```python

```

```python
yield_table = pd.DataFrame([
    ["a", "?", "a"] + [x*15 for x in range(0,len(age_classes))],
    ["a", "?", "b"] + [x*15 for x in range(0,len(age_classes))],
    ["b", "?", "b"] + [x*15 for x in range(0,len(age_classes))]
])

```

```python
yield_table
```

```python
config = {
    "species":{
        "species_classifier": "classifier1",
        "species_mapping":[
            {"user_species": "a", "default_species": "Spruce"},
            {"user_species": "b", "default_species": "Oak"}
        ]
    }
}
sit_mapping = SITMapping(config, cbm_defaults_ref)
default_species_map = sit_mapping.get_species_map(classifiers, classifier_values)
```

```python
parsed_yield_table = sit_yield_parser.parse(yield_table, classifiers, classifier_values, age_classes, default_species_map)
```

```python
parsed_yield_table
```

```python
num_eligibility_cols = len(sit_format.get_disturbance_eligibility_columns(0))
event = {"classifier_set": ["a","?"],
         "age_eligibility": ["False", -1,-1,-1,-1],
         "eligibility": [-1] * num_eligibility_cols,
         "target": [1.0, "1", "A", 100, "dist1", 2, 100]
        }
disturbance_event_table = pd.DataFrame([
    event["classifier_set"] + event["age_eligibility"] + event["eligibility"] + event["target"]
])
disturbance_event_table
```

```python

```

```python
disturbance_events = sit_disturbance_event_parser.parse(
    disturbance_event_table, classifiers, classifier_values,
    classifier_aggregates, disturbance_types, age_classes)

```

```python
disturbance_events
```

```python
transition = {"classifier_set_src": ["a","agg1"],
              "age_eligibility": ["TRUE", "age_1", "age_5","age_1", "age_5"],
              "disturbance_type": ["dist1"],
              "classifier_set_dest": ["b","?"],
              "post_transition": [0,-1,49]}
transition_table = pd.DataFrame([
    transition["classifier_set_src"] + transition["age_eligibility"] + 
    transition["disturbance_type"] + transition["classifier_set_dest"] +
    transition["post_transition"]
] +
[
    transition["classifier_set_src"] + transition["age_eligibility"] + 
    transition["disturbance_type"] + transition["classifier_set_dest"] +
    transition["post_transition"]
])
transition_table
```

```python
transitions = sit_transition_rule_parser.parse(transition_table, classifiers, classifier_values,
    classifier_aggregates, disturbance_types, age_classes)
```

```python
transitions
```

```python
config = {
    "disturbance_types":[
        {"user_dist_type": "fire", "default_dist_type": "Wildfire"}
    ]        
}



```

```python

```

```python
cbm = sit_cbm_factory.initialize_cbm(
    db_path, dll_path, parsed_yield_table, classifiers, classifier_values, age_classes, cbm_defaults_ref)

```

```python
cbm_defaults_ref
```

```python
cbm.step
```

```python
inventory
```

```python
classifiers
```

```python
classifier_values
```

```python

```

```python

```

```python
classifier_config = sit_cbm_factory.get_classifiers(classifiers, classifier_values)
```

```python
classifier_config
```

```python
classifier_ids = [(x["id"],x["name"]) for x in classifier_config["classifiers"]]

```

```python
classifier_ids
```

```python
classifier_value_id_lookups = {}

for identifier, name in classifier_ids:
    classifier_value_id_lookups[name] = {x["value"]: x["id"] for x in classifier_config["classifier_values"] if x["classifier_id"]==identifier}
```

```python
classifier_value_id_lookups
```

```python
classifiers_result = pd.DataFrame(
    data={
        name: inventory[name].map(classifier_value_id_lookups[name]) 
        for name in list(classifiers.name)},
    columns=list(classifiers.name))
```

```python
classifiers_result
```

```python
cbm_defaults_ref.land_class_ref[0]["code"]
```

```python
inventory_result = pd.DataFrame(
    data={
        "age": inventory.age,
        "spatial_unit": 42,
        "afforestation_pre_type_id": 0,
        "area": inventory.area,
        "delay": inventory.delay,
        "land_class": inventory.land_class.map(cbm_defaults_ref.get_land_class_id),
        "historical_disturbance_type": 1,
        "last_pass_disturbance_type": 1,
    })
```

```python
inventory_result
```

```python
from libcbm.model.cbm import cbm_variables
n_stands = len(inventory_result)
pools = cbm_variables.initialize_pools(n_stands, cbm_defaults_ref.get_pools())
flux_indicators = cbm_variables.initialize_flux(n_stands, cbm_defaults_ref.get_flux_indicators())
spinup_params = cbm_variables.initialize_spinup_parameters(n_stands)
spinup_variables = cbm_variables.initialize_spinup_variables(n_stands)
cbm_params = cbm_variables.initialize_cbm_parameters(n_stands)
cbm_state = cbm_variables.initialize_cbm_state_variables(n_stands)
cbm_inventory = cbm_variables.initialize_inventory(
    classifiers=classifiers_result,
    inventory=inventory_result)

```

```python
inventory_result = np.ascontiguousarray(inventory_result)
inventory_result.flags
```

```python
cbm.spinup(cbm_inventory, pools, spinup_variables, spinup_params)
```

```python
pools
```

```python
cbm.init(cbm_inventory, pools, cbm_state)
```

```python
cbm_state
```

```python
cbm.step(cbm_inventory, pools, flux_indicators, cbm_state, cbm_params)
```

```python
cbm_state
```

```python
cbm.step(cbm_inventory, pools, flux_indicators, cbm_state, cbm_params)
```

```python
cbm_params
```

```python
pools
```

```python

```
