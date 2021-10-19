import pandas as pd

from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.stand_cbm_factory import StandCBMFactory


def test_integration():

    classifiers = {
        "c1": ["c1_v1"]
    }
    merch_volumes = [
        {
            "classifier_set": ["c1_v1"],
            "merch_volumes": [{
                "species": "Spruce",
                "age_volume_pairs": [
                    [0, 0],
                    [50, 100],
                    [100, 150],
                    [150, 200],
                ]
            }]
        }
    ]

    inventory = pd.DataFrame(
        columns=[
            "c1", "admin_boundary", "eco_boundary", "age", "area", "delay",
            "landclass", "afforestation_pre_type",
            "historic_disturbance_type", "last_pass_disturbance_type"],
        data=[
            ["c1_v1", "British Columbia", "Pacific Maritime", 15, 1.0, 0,
             "UNFCCC_FL_FL", None, "Wildfire", "Wildfire"]])

    cbm_factory = StandCBMFactory(classifiers, merch_volumes)

    csets, inv = cbm_factory.prepare_inventory(inventory)
    n_steps = 10
    n_stands = len(inv.index)
    with cbm_factory.initialize_cbm() as cbm:

        spinup_results, spinup_reporting_func = \
            cbm_simulator.create_in_memory_reporting_func(density=True)
        cbm_results, cbm_reporting_func = \
            cbm_simulator.create_in_memory_reporting_func(
                classifier_map={1: "c1_v1"}, disturbance_type_map={1: "fires"})
        cbm_simulator.simulate(
            cbm,
            n_steps=n_steps,
            classifiers=csets,
            inventory=inv,
            pre_dynamics_func=lambda t, cbm_vars: cbm_vars,
            reporting_func=cbm_reporting_func,
            spinup_params=cbm_variables.initialize_spinup_parameters(
                n_stands, 50, 5, 5, -1),
            spinup_reporting_func=spinup_reporting_func)
        assert len(cbm_results.pools.index) == (n_steps + 1) * n_stands
