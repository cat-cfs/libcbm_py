import numpy as np
import pandas as pd
from libcbm.model import model_definition


def test_integration():
    pool_def = {
        "Input": 0,
        "WoodyBiomass": 1,
        "Foliage": 2,
        "SlowDOM": 3,
        "MediumDOM": 4,
        "FastDOM": 5,
        "CO2": 6,
        "Products": 7,
    }

    processes = {"GrowthAndMortality": 0, "Decay": 1, "Disturbance": 2}

    flux_indicators = [
        {
            "name": "NPP",
            "process_id": processes["GrowthAndMortality"],
            "source_pools": [pool_def["Input"], pool_def["Foliage"]],
            "sink_pools": [
                pool_def["WoodyBiomass"],
                pool_def["Foliage"],
                pool_def["FastDOM"],
            ],
        },
        {
            "name": "DecayEmissions",
            "process_id": processes["Decay"],
            "source_pools": [
                pool_def["SlowDOM"],
                pool_def["MediumDOM"],
                pool_def["FastDOM"],
            ],
            "sink_pools": [pool_def["CO2"]],
        },
        {
            "name": "DisturbanceEmissions",
            "process_id": processes["Disturbance"],
            "source_pools": [
                pool_def["WoodyBiomass"],
                pool_def["Foliage"],
                pool_def["SlowDOM"],
                pool_def["MediumDOM"],
                pool_def["FastDOM"],
            ],
            "sink_pools": [pool_def["CO2"]],
        },
        {
            "name": "HarvestProduction",
            "process_id": processes["Disturbance"],
            "source_pools": [
                pool_def["WoodyBiomass"],
                pool_def["Foliage"],
                pool_def["MediumDOM"],
            ],
            "sink_pools": [pool_def["Products"]],
        },
    ]

    def weibull_cumulative(x, k=2.3, y=1):
        c = np.power(x / y, k)
        return 1 - np.exp(-c)

    def get_npp_matrix(model, age):
        # creates NPP flows based on an age passed to the cumulative
        # weibull distribution
        n_stands = age.shape[0]
        npp = weibull_cumulative((age + 1) / 100.0) - weibull_cumulative(
            age / 100.0
        )
        op = model.create_operation(
            matrices=[
                ["Input", "WoodyBiomass", npp],
                ["Input", "Foliage", npp / 10.0],
            ],
            fmt="repeating_coordinates",
        )

        op.set_matrix_index(np.arange(0, n_stands))
        return op

    pd.DataFrame(
        [weibull_cumulative(x, 5, 1) for x in np.arange(0, 2.5, 0.01)]
    ).plot()

    def get_mortality_matrix(model, n_stands):

        op = model.create_operation(
            matrices=[
                ["WoodyBiomass", "WoodyBiomass", 1.0],
                ["WoodyBiomass", "MediumDOM", 0.01],
                ["Foliage", "Foliage", 1.0],
                ["Foliage", "FastDOM", 0.95],
            ],
            fmt="repeating_coordinates",
        )
        # set every stand to point at the 0th matrix:
        # they all share the same simple mortality matrix
        op.set_matrix_index(np.full(n_stands, 0))
        return op

    def get_decay_matrix(model, n_stands):
        op = model.create_operation(
            matrices=[
                ["SlowDOM", "SlowDOM", 0.99],
                ["SlowDOM", "CO2", 0.01],
                ["MediumDOM", "MediumDOM", 0.85],
                ["MediumDOM", "SlowDOM", 0.10],
                ["MediumDOM", "CO2", 0.05],
                ["FastDOM", "FastDOM", 0.65],
                ["FastDOM", "MediumDOM", 0.25],
                ["FastDOM", "CO2", 0.10],
            ],
            fmt="repeating_coordinates",
        )
        op.set_matrix_index(np.full(n_stands, 0))
        return op

    disturbance_type_ids = {"none": 0, "fire": 1, "harvest": 2}

    def get_disturbance_matrix(model, disturbance_types):

        no_disturbance = []
        fire_matrix = [
            ["WoodyBiomass", "WoodyBiomass", 0.0],
            ["WoodyBiomass", "CO2", 0.85],
            ["WoodyBiomass", "MediumDOM", 0.15],
            ["Foliage", "Foliage", 0.0],
            ["Foliage", "CO2", 0.95],
            ["Foliage", "FastDOM", 0.05],
        ]
        harvest_matrix = [
            ["WoodyBiomass", "WoodyBiomass", 0.0],
            ["WoodyBiomass", "Products", 0.85],
            ["WoodyBiomass", "MediumDOM", 0.15],
            ["Foliage", "Foliage", 0.0],
            ["Foliage", "FastDOM", 1.0],
        ]

        op = model.create_operation(
            matrices=[no_disturbance, fire_matrix, harvest_matrix],
            fmt="matrix_list",
        )
        op.set_matrix_index(disturbance_types)
        return op

    with model_definition.create_model(pool_def, flux_indicators) as model:

        output_processor = model.create_output_processor()
        n_stands = 2
        model_vars = model.allocate_model_vars(n_stands)
        model_vars.pools["Input"].assign(1.0)

        stand_age = np.full(n_stands, 0)

        for t in range(0, 1000):

            # add some simplistic disturbance scheduling

            if (t % 150) == 0:
                disturbance_types = np.full(
                    n_stands, disturbance_type_ids["fire"]
                )
            elif t == 950:
                disturbance_types = np.full(
                    n_stands, disturbance_type_ids["harvest"]
                )
            else:
                disturbance_types = np.full(
                    n_stands, disturbance_type_ids["none"]
                )

            # reset flux at start of every time step
            model_vars.flux.zero()

            # prepare the matrix operations
            operations = [
                get_disturbance_matrix(model, disturbance_types),
                get_npp_matrix(model, stand_age),
                get_mortality_matrix(model, n_stands),
                get_decay_matrix(model, n_stands),
            ]

            # associate each above operation with a flux indicator category
            op_processes = [
                processes["Disturbance"],
                processes["GrowthAndMortality"],
                processes["GrowthAndMortality"],
                processes["Decay"],
            ]

            # enabled can be used to disable(0)/enable(1) dynamics per index
            model_vars.enabled.assign(1)

            model.compute(model_vars, operations, op_processes)
            for op in operations:
                op.dispose()
            output_processor.append_results(t, model_vars)
            stand_age[disturbance_types != 0] = 0
            stand_age += 1

    output_processor.pools.columns

    output_processor.pools.to_pandas()[
        [
            "timestep",
            "WoodyBiomass",
            "Foliage",
            "SlowDOM",
            "MediumDOM",
            "FastDOM",
        ]
    ].groupby("timestep").sum().plot(figsize=(15, 10))

    output_processor.flux.to_pandas()[
        [
            "timestep",
            "NPP",
            "DecayEmissions",
            "DisturbanceEmissions",
            "HarvestProduction",
        ]
    ].groupby("timestep").sum().plot(figsize=(15, 10))
