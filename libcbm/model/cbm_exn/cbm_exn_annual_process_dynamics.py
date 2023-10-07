from typing import Union
import numpy as np


def net_growth(
    growth_info: dict[str, np.ndarray],
) -> list:
    matrices = [
        ["Input", "Merch", growth_info["merch_inc"] * 0.5],
        ["Input", "Other", growth_info["other_inc"] * 0.5],
        ["Input", "Foliage", growth_info["foliage_inc"] * 0.5],
        ["Input", "CoarseRoots", growth_info["coarse_root_inc"] * 0.5],
        ["Input", "FineRoots", growth_info["fine_root_inc"] * 0.5],
    ]
    return matrices


def overmature_decline(
    growth_info: dict[str, np.ndarray],
) -> list:
    matrices = [
        ["Merch", "StemSnag", growth_info["merch_to_stem_snag_prop"]],
        ["Other", "BranchSnag", growth_info["other_to_branch_snag_prop"]],
        [
            "Other",
            "AboveGroundFastSoil",
            growth_info["other_to_ag_fast_prop"],
        ],
        [
            "Foliage",
            "AboveGroundVeryFastSoil",
            growth_info["foliage_to_ag_fast_prop"],
        ],
        [
            "CoarseRoots",
            "AboveGroundFastSoil",
            growth_info["coarse_root_to_ag_fast_prop"],
        ],
        [
            "CoarseRoots",
            "BelowGroundFastSoil",
            growth_info["coarse_root_to_bg_fast_prop"],
        ],
        [
            "FineRoots",
            "AboveGroundVeryFastSoil",
            growth_info["fine_root_to_ag_vfast_prop"],
        ],
        [
            "FineRoots",
            "BelowGroundVeryFastSoil",
            growth_info["fine_root_to_bg_vfast_prop"],
        ],
    ]
    return matrices


def snag_turnover(rates: dict[str, np.ndarray]) -> list:
    matrices = [
        ["StemSnag", "StemSnag", 1 - rates["StemSnag"]],
        ["StemSnag", "MediumSoil", rates["StemSnag"]],
        ["BranchSnag", "BranchSnag", 1 - rates["BranchSnag"]],
        ["BranchSnag", "AboveGroundFastSoil", rates["BranchSnag"]],
    ]
    return matrices


def biomass_turnover(rates: dict[str, np.ndarray]) -> list:
    matrices = [
        [
            "Merch",
            "StemSnag",
            rates["StemAnnualTurnoverRate"],
        ],
        [
            "Foliage",
            "AboveGroundVeryFastSoil",
            rates["FoliageFallRate"],
        ],
        [
            "Other",
            "BranchSnag",
            rates["OtherToBranchSnagSplit"] * rates["BranchTurnoverRate"],
        ],
        [
            "Other",
            "AboveGroundFastSoil",
            (1 - rates["OtherToBranchSnagSplit"])
            * rates["BranchTurnoverRate"],
        ],
        [
            "CoarseRoots",
            "AboveGroundFastSoil",
            rates["CoarseRootAGSplit"] * rates["CoarseRootTurnProp"],
        ],
        [
            "CoarseRoots",
            "BelowGroundFastSoil",
            (1 - rates["CoarseRootAGSplit"]) * rates["CoarseRootTurnProp"],
        ],
        [
            "FineRoots",
            "AboveGroundVeryFastSoil",
            rates["FineRootAGSplit"] * rates["FineRootTurnProp"],
        ],
        [
            "FineRoots",
            "BelowGroundVeryFastSoil",
            (1 - rates["FineRootAGSplit"]) * rates["FineRootTurnProp"],
        ],
    ]
    return matrices


def compute_decay_rate(
    mean_annual_temp: np.ndarray,
    base_decay_rate: Union[np.ndarray, float],
    q10: Union[np.ndarray, float],
    tref: Union[np.ndarray, float],
    max: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Compute a CBM-CFS3 DOM pool specific decay rate based on mean annual
    temperature and other parameters.

    Args:
        mean_annual_temp (np.ndarray): mean annual temperature (deg C)
        base_decay_rate (np.ndarray): base decay rate for DOM pool
        q10 (np.ndarray): Q10 decay rate parameter
        tref (np.ndarray): reference temperature decay rate parameter
        max (np.ndarray): maximum decay rate

    Returns:
        np.ndarray: proportional decay rates
    """
    return np.minimum(
        base_decay_rate
        * np.exp((mean_annual_temp - tref) * np.log(q10) * 0.1),
        max,
    )


def dom_decay(
    mean_annual_temp: np.ndarray, decay_parameters: dict[str, dict[str, float]]
) -> list:
    dom_pools = [
        "AboveGroundVeryFastSoil",
        "BelowGroundVeryFastSoil",
        "AboveGroundFastSoil",
        "BelowGroundFastSoil",
        "MediumSoil",
        "StemSnag",
        "BranchSnag",
    ]
    dom_pool_flows = {
        "AboveGroundVeryFastSoil": "AboveGroundSlowSoil",
        "BelowGroundVeryFastSoil": "BelowGroundSlowSoil",
        "AboveGroundFastSoil": "AboveGroundSlowSoil",
        "BelowGroundFastSoil": "BelowGroundSlowSoil",
        "MediumSoil": "AboveGroundSlowSoil",
        "StemSnag": "AboveGroundSlowSoil",
        "BranchSnag": "AboveGroundSlowSoil",
    }
    matrix_data = []
    for dom_pool in dom_pools:
        decay_parameter = decay_parameters[dom_pool]
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["reference_temp"],
            max=decay_parameter["max_rate"],
        )
        matrix_data.append([dom_pool, dom_pool, 1 - decay_rate])
        matrix_data.append(
            [
                dom_pool,
                dom_pool_flows[dom_pool],
                decay_rate * (1 - prop_to_atmosphere),
            ]
        )
        matrix_data.append([dom_pool, "CO2", decay_rate * prop_to_atmosphere])

    return matrix_data


def slow_decay(
    mean_annual_temp: np.ndarray, decay_parameters: dict[str, dict[str, float]]
) -> list:
    matrix_data = []
    for dom_pool in ["AboveGroundSlowSoil", "BelowGroundSlowSoil"]:
        decay_parameter = decay_parameters[dom_pool]
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["reference_temp"],
            max=decay_parameter["max_rate"],
        )
        matrix_data.append([dom_pool, dom_pool, 1 - decay_rate])
        matrix_data.append(
            [
                dom_pool,
                "CO2",
                decay_rate * prop_to_atmosphere,
            ]
        )

    return matrix_data


def slow_mixing(rate: float) -> list:
    return [
        ["AboveGroundSlowSoil", "BelowGroundSlowSoil", rate],
        ["AboveGroundSlowSoil", "AboveGroundSlowSoil", 1 - rate],
    ]
