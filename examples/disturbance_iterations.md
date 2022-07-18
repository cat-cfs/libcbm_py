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

## Imports

```python
import os
import pandas as pd
%matplotlib inline
```

```python
from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm import resources
```

## Setup

```python
data_path       = os.path.join(resources.get_test_resources_dir(), "cbm3_tutorial2")
config_path     = os.path.join(data_path, "sit_config.json")
sit_events_path = os.path.join(data_path, "disturbance_events.csv")
sit             = sit_cbm_factory.load_sit(config_path)

classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
```

#### Create a template `sit_events` for dynamic `sit_events`

```python
event_template = pd.read_csv(sit_events_path).iloc[[0]]
display(event_template)
```

## Dynamic harvest processing


### Compute disturbance production method

<!-- #region -->
The `compute_disturbance_production` will be used in the following cells. See the following example for usage:

**Example**

```python
production_df = cbm.compute_disturbance_production(cbm_vars)
```

This does not alter the values stored in cbm_vars in any way. By default, this will compute the total disturbance production (flows to products pools) using:

* The current pools stored in cbm_vars.pool.
* The disturbances assigned to `cbm_vars.parameters.disturbance_type`.

The result is returned as a pandas.DataFrame along the same index as the values in cbm_vars.

It is also possible to compute disturbance production for particular disturbance types or subsets of the cbm_vars values. See: `libcbm.model.cbm.cbm_model.CBM.compute_disturbance_production`.
<!-- #endregion -->

```python
class DynamicHarvestProcessor:
    """
    Class that dynamically generates dynamic events using an
    event template to meet the specified production target.
    """

    def __init__(self, sit, cbm, production_target, event_template):
        self._event_template = event_template
        self._production_target = production_target
        self._sit = sit
        self._cbm = cbm
        self._base_processor = \
            sit_cbm_factory.create_sit_rule_based_processor(
                self._sit, self._cbm)
        self._dynamic_stats_list = []
        self._base_production_totals = []

    def get_base_process_stats(self):
        """Gets the stats for all disturbances in `sit.sit_data.disturbance_events`."""
        stats_df = pd.concat(
            self._base_processor.sit_event_stats_by_timestep.values())
        return stats_df.merge(
            self._sit.sit_data.disturbance_events,
            left_on="sit_event_index",
            right_index=True)

    def get_base_production_totals(self):
        return pd.DataFrame(
            columns=["timestep", "total_production"],
            data=self._base_production_totals)

    def get_dynamic_process_stats(self):
        return pd.concat(self._dynamic_stats_list).reset_index(drop=True)

    def pre_dynamics_func(self, timestep, cbm_vars):
        """
        Use a production target (tonnes C) to apply across all years
        this will be partially met by the base tutorial2 events,
        then fully met by a second dynamically generated event.
        """
        cbm_vars = self._base_processor.pre_dynamics_func(timestep, cbm_vars)

        # Compute the total production resulting from the sit_events
        # bundled in the tutorial2 dataset
        production_df = self._cbm.compute_disturbance_production(cbm_vars, density=False)
        total_production = production_df["Total"].sum()
        self._base_production_totals.append([timestep, total_production])

        remaining_production = self._production_target - total_production
        if remaining_production <= 0:
            # Target already met
            return cbm_vars

        dynamic_event = self._event_template.reset_index(drop=True)
        dynamic_event["disturbance_year"] = timestep
        dynamic_event["target_type"] = "M"
        dynamic_event["target"] = remaining_production

        # See the documentation:
        # `libcbm.input.sit.sit_cbm_factory.create_sit_rule_based_processor`
        dynamic_processor = sit_cbm_factory.create_sit_rule_based_processor(
            self._sit, self._cbm, reset_parameters=False,
            sit_events=dynamic_event)

        cbm_vars = dynamic_processor.pre_dynamics_func(timestep, cbm_vars)
        self._dynamic_stats_list.append(
            dynamic_processor.sit_event_stats_by_timestep[timestep].merge(
                dynamic_event, left_on="sit_event_index", right_index=True))
        return cbm_vars

```

## Simulation

```python
cbm_output = CBMOutput()
with sit_cbm_factory.initialize_cbm(sit) as cbm:

    dynamic_harvest_processor = DynamicHarvestProcessor(
        sit, cbm, production_target=4800,
        event_template=event_template)

    cbm_simulator.simulate(
        cbm,
        n_steps           = 200,
        classifiers       = classifiers,
        inventory         = inventory,
        pre_dynamics_func = dynamic_harvest_processor.pre_dynamics_func,
        reporting_func    = cbm_output.append_simulation_result
    )
```

## Results

```python
base_process_stats = dynamic_harvest_processor.get_base_process_stats()
display(base_process_stats)
```

```python
base_production_totals = dynamic_harvest_processor.get_base_production_totals()
display(base_production_totals)
```

```python
dynamic_processor_stats = dynamic_harvest_processor.get_dynamic_process_stats()
display(dynamic_processor_stats)
```

## Summary

The base harvest, based on the CBM tutorial2 SIT_events dataset resulted in a disturbance production of less than 5000 tonnes C for all timesteps. The dynamic processor made up the difference of 4800 - base production, and the result was that for all time steps the 4800 tC target was met exactly.

```python
flux_results = pd.DataFrame({
    "timestep": cbm_output.flux.to_pandas().timestep,
    "flux_production_total": cbm_output.flux.to_pandas()[[
        'DisturbanceSoftProduction',
        'DisturbanceHardProduction',
        'DisturbanceDOMProduction']].sum(axis=1)}).groupby("timestep").sum()


dynamic_results = dynamic_processor_stats[["disturbance_year", "total_achieved"]].groupby("disturbance_year").sum()

summary = pd.DataFrame({
    "flux_indicator_total":    flux_results["flux_production_total"],
    "base_processor_total":    base_production_totals.set_index("timestep")["total_production"],
    "dynamic_processor_total": dynamic_results["total_achieved"]
})

display(summary)
```

```python
summary.plot(figsize=(15,10))
```

```python

```
