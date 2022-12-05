import numpy as np
import pandas as pd
from libcbm.model.model_definition import model
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.model_definition.cbm_variables import CBMVariables


def test_integration():
    pool_def = {
        "Input",
        "WoodyBiomass",
        "Foliage",
        "SlowDOM",
        "MediumDOM",
        "FastDOM",
        "CO2",
        "Products",
    }

    processes = {"GrowthAndMortality": 0, "Decay": 1, "Disturbance": 2}

    flux_indicators = [
        {
            "name": "NPP",
            "process": processes["GrowthAndMortality"],
            "source_pools": ["Input", "Foliage"],
            "sink_pools": [
                "WoodyBiomass",
                "Foliage",
                "FastDOM",
            ],
        },
        {
            "name": "DecayEmissions",
            "process": processes["Decay"],
            "source_pools": [
                "SlowDOM",
                "MediumDOM",
                "FastDOM",
            ],
            "sink_pools": ["CO2"],
        },
        {
            "name": "DisturbanceEmissions",
            "process": processes["Disturbance"],
            "source_pools": [
                "WoodyBiomass",
                "Foliage",
                "SlowDOM",
                "MediumDOM",
                "FastDOM",
            ],
            "sink_pools": ["CO2"],
        },
        {
            "name": "HarvestProduction",
            "process": processes["Disturbance"],
            "source_pools": [
                "WoodyBiomass",
                "Foliage",
                "MediumDOM",
            ],
            "sink_pools": ["Products"],
        },
    ]

    def weibull_cumulative(x, k=2.3, y=1):
        c = np.power(x / y, k)
        return 1 - np.exp(-c)

    def get_npp_matrix(cbm_model: CBMModel, age: np.ndarray):
        # creates NPP flows based on an age passed to the cumulative
        # weibull distribution
        n_stands = age.shape[0]
        npp = weibull_cumulative((age + 1) / 100.0) - weibull_cumulative(
            age / 100.0
        )
        op = cbm_model.create_operation(
            matrices=[
                ["Input", "WoodyBiomass", npp],
                ["Input", "Foliage", npp / 10.0],
            ],
            fmt="repeating_coordinates",
            process_id=processes["GrowthAndMortality"]
        )

        op.set_op(np.arange(0, n_stands))
        return op

    pd.DataFrame(
        [weibull_cumulative(x, 5, 1) for x in np.arange(0, 2.5, 0.01)]
    ).plot()

    def get_mortality_matrix(cbm_model: CBMModel, n_stands: int):

        op = cbm_model.create_operation(
            matrices=[
                ["WoodyBiomass", "WoodyBiomass", 1.0],
                ["WoodyBiomass", "MediumDOM", 0.01],
                ["Foliage", "Foliage", 1.0],
                ["Foliage", "FastDOM", 0.95],
            ],
            fmt="repeating_coordinates",
            process_id=processes["GrowthAndMortality"]
        )
        # set every stand to point at the 0th matrix:
        # they all share the same simple mortality matrix
        op.set_op(np.full(n_stands, 0))
        return op

    def get_decay_matrix(cbm_model: CBMModel, n_stands: int):
        op = cbm_model.create_operation(
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
            process_id=processes["Decay"]
        )
        op.set_op(np.full(n_stands, 0))
        return op

    disturbance_type_ids = {"none": 0, "fire": 1, "harvest": 2}

    def get_disturbance_matrix(
        cbm_model: CBMModel, disturbance_types: np.ndarray
    ):

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

        op = cbm_model.create_operation(
            matrices=[no_disturbance, fire_matrix, harvest_matrix],
            fmt="matrix_list",
            process_id=processes["Disturbance"],
        )
        op.set_op(disturbance_types)
        return op

    with model.initialize(pool_def, flux_indicators) as model_handle:

        output_processor = ModelOutputProcessor()
        n_stands = 2
        model_vars = CBMVariables.from_pandas(
            {
                "pools": pd.DataFrame(
                    columns=model_handle.pool_names,
                    data={
                        p: np.zeros(n_stands) for p in model_handle.pool_names
                    },
                ),
                "flux": pd.DataFrame(
                    columns=model_handle.flux_names,
                    data={
                        f: np.zeros(n_stands) for f in model_handle.flux_names
                    },
                ),
                "state": pd.DataFrame(
                    columns=["enabled"], data=np.ones(n_stands, dtype="int")
                ),
            }
        )
        model_vars["pools"]["Input"].assign(1.0)

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
            model_vars["flux"].zero()

            # prepare the matrix operations
            operations = [
                get_disturbance_matrix(model_handle, disturbance_types),
                get_npp_matrix(model_handle, stand_age),
                get_mortality_matrix(model_handle, n_stands),
                get_decay_matrix(model_handle, n_stands),
            ]

            # enabled can be used to disable(0)/enable(1) dynamics per index
            model_vars["state"]["enabled"].assign(1)

            model_handle.compute(model_vars, operations)
            for op in operations:
                op.dispose()
            output_processor.append_results(t, model_vars.get_collection())
            stand_age[disturbance_types != 0] = 0
            stand_age += 1

    output_pools = output_processor.get_results()["pools"]
    output_pools.columns
    output_flux = output_processor.get_results()["flux"]
    output_pools.to_pandas()[
        [
            "timestep",
            "WoodyBiomass",
            "Foliage",
            "SlowDOM",
            "MediumDOM",
            "FastDOM",
        ]
    ].groupby("timestep").sum().plot(figsize=(15, 10))

    output_flux.to_pandas()[
        [
            "timestep",
            "NPP",
            "DecayEmissions",
            "DisturbanceEmissions",
            "HarvestProduction",
        ]
    ].groupby("timestep").sum().plot(figsize=(15, 10))
