import pandas as pd
from libcbm.storage import dataframe
from libcbm.storage import series
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.stand_cbm_factory import StandCBMFactory
from libcbm.model.cbm.cbm_output import CBMOutput


def test_integration():

    classifiers = {
        "c1": ["c1_v1"],
        "c2": ["c2_v1"],
    }
    merch_volumes = [
        {
            "classifier_set": ["c1_v1", "?"],
            "merch_volumes": [
                {
                    "species": "Spruce",
                    "age_volume_pairs": [
                        [0, 0],
                        [50, 100],
                        [100, 150],
                        [150, 200],
                    ],
                }
            ],
        }
    ]

    cbm_factory = StandCBMFactory(classifiers, merch_volumes)

    n_steps = 10
    return_interval = 50
    n_rotations = 5
    age = 15
    inventory = dataframe.from_pandas(
        pd.DataFrame(
            columns=[
                "c1",
                "c2",
                "admin_boundary",
                "eco_boundary",
                "age",
                "area",
                "delay",
                "land_class",
                "afforestation_pre_type",
                "historic_disturbance_type",
                "last_pass_disturbance_type",
            ],
            data=[
                [
                    "c1_v1",
                    "c2_v1",
                    "British Columbia",
                    "Pacific Maritime",
                    age,
                    1.0,
                    0,
                    "UNFCCC_FL_R_FL",
                    None,
                    "Wildfire",
                    "Wildfire",
                ]
            ],
        )
    )
    csets, inv = cbm_factory.prepare_inventory(inventory)
    n_stands = inv.n_rows
    with cbm_factory.initialize_cbm() as cbm:

        spinup_results = CBMOutput(density=True)

        cbm_results = CBMOutput(
            classifier_map=cbm_factory.classifier_value_names,
            disturbance_type_map=cbm_factory.disturbance_types,
        )

        cbm_simulator.simulate(
            cbm,
            n_steps=n_steps,
            classifiers=csets,
            inventory=inv,
            pre_dynamics_func=lambda t, cbm_vars: cbm_vars,
            reporting_func=cbm_results.append_simulation_result,
            spinup_params=cbm_variables.initialize_spinup_parameters(
                n_stands,
                inventory.backend_type,
                series.allocate(
                    "return_interval",
                    n_stands,
                    return_interval,
                    "int32",
                    inventory.backend_type,
                ),
                series.allocate(
                    "min_rotations",
                    n_stands,
                    n_rotations,
                    "int32",
                    inventory.backend_type,
                ),
                series.allocate(
                    "max_rotations",
                    n_stands,
                    n_rotations,
                    "int32",
                    inventory.backend_type,
                ),
                series.allocate(
                    "mean_annual_temp",
                    n_stands,
                    -1,
                    "float",
                    inventory.backend_type,
                ),
            ),
            spinup_reporting_func=spinup_results.append_simulation_result,
        )
        assert cbm_results.pools.n_rows == (n_steps + 1) * n_stands
        assert (
            spinup_results.pools.n_rows
            == (n_rotations * return_interval) + age - 1
        )
