from enum import IntEnum


class Pool(IntEnum):
    """
    Enumeration of Moss C pools values
    """

    Input = 0
    FeatherMossLive = 1
    SphagnumMossLive = 2
    FeatherMossFast = 3
    SphagnumMossFast = 4
    FeatherMossSlow = 5
    SphagnumMossSlow = 6
    CO2 = 7
    CH4 = 8
    CO = 9
    Products = 10


ANNUAL_PROCESSES = 1

DISTURBANCE_PROCESS = 2

BIOMASS_POOLS = [
    Pool.FeatherMossLive,
    Pool.SphagnumMossLive,
]


DOM_POOLS = [
    Pool.FeatherMossFast,
    Pool.SphagnumMossFast,
    Pool.FeatherMossSlow,
    Pool.SphagnumMossSlow,
]

EMISSIONS_POOLS = [Pool.CO2, Pool.CH4, Pool.CO]

ECOSYSTEM_POOLS = BIOMASS_POOLS + DOM_POOLS

FLUX_INDICATORS = [
    {
        "name": "NPPFeatherMoss",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.Input],
        "sink_pools": [Pool.FeatherMossLive],
    },
    {
        "name": "NPPSphagnumMoss",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.Input],
        "sink_pools": [Pool.SphagnumMossLive],
    },
    {
        "name": "DisturbanceCO2Production",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": ECOSYSTEM_POOLS,
        "sink_pools": [Pool.CO2],
    },
    {
        "name": "DisturbanceCH4Production",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": ECOSYSTEM_POOLS,
        "sink_pools": [Pool.CH4],
    },
    {
        "name": "DisturbanceCOProduction",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": ECOSYSTEM_POOLS,
        "sink_pools": [Pool.CO],
    },
    {
        "name": "DisturbanceBioCO2Emission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": BIOMASS_POOLS,
        "sink_pools": [Pool.CO],
    },
    {
        "name": "DisturbanceBioCH4Emission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": BIOMASS_POOLS,
        "sink_pools": [Pool.CH4],
    },
    {
        "name": "DisturbanceBioCOEmission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": BIOMASS_POOLS,
        "sink_pools": [Pool.CO],
    },
    {
        "name": "DecayDOMCO2Emission",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": DOM_POOLS,
        "sink_pools": [Pool.CO2],
    },
    {
        "name": "DisturbanceBioProduction",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": BIOMASS_POOLS,
        "sink_pools": [Pool.Products],
    },
    {
        "name": "DisturbanceDOMProduction",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": DOM_POOLS,
        "sink_pools": [Pool.Products],
    },
    {
        "name": "TurnoverFeatherMoss",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.FeatherMossLive],
        "sink_pools": [Pool.FeatherMossFast, Pool.FeatherMossSlow],
    },
    {
        "name": "TurnoverSphagnumMoss",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.FeatherMossLive],
        "sink_pools": [Pool.FeatherMossFast, Pool.FeatherMossSlow],
    },
    {
        "name": "DecayFeatherMossFastToAir",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.FeatherMossFast],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DecaySphagnumMossFastToAir",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.SphagnumMossFast],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DecayFeatherMossSlowToAir",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.FeatherMossSlow],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DecaySphagnumMossSlowToAir",
        "process_id": ANNUAL_PROCESSES,
        "source_pools": [Pool.SphagnumMossSlow],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceFeatherMossToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.FeatherMossLive],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceSphagnumMossToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.SphagnumMossLive],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceDOMCO2Emission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": DOM_POOLS,
        "sink_pools": [Pool.CO2],
    },
    {
        "name": "DisturbanceDOMCH4Emission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": DOM_POOLS,
        "sink_pools": [Pool.CH4],
    },
    {
        "name": "DisturbanceDOMCOEmission",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": DOM_POOLS,
        "sink_pools": [Pool.CO],
    },
    {
        "name": "DisturbanceFeatherMossLitterInput",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.FeatherMossLive],
        "sink_pools": [Pool.FeatherMossFast, Pool.FeatherMossSlow],
    },
    {
        "name": "DisturbanceSphagnumMossLitterInput",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.SphagnumMossLive],
        "sink_pools": [Pool.SphagnumMossFast, Pool.SphagnumMossSlow],
    },
    {
        "name": "DisturbanceFeatherMossFastToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.FeatherMossFast],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceSphagnumMossFastToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.SphagnumMossFast],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceFeatherMossSlowToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.FeatherMossSlow],
        "sink_pools": EMISSIONS_POOLS,
    },
    {
        "name": "DisturbanceSphagnumMossSlowToAir",
        "process_id": DISTURBANCE_PROCESS,
        "source_pools": [Pool.SphagnumMossSlow],
        "sink_pools": EMISSIONS_POOLS,
    },
]
