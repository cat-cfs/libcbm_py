from typing import Union
import numpy as np
import pandas as pd


def net_growth(
    growth_info: dict[str, np.ndarray],
) -> pd.DataFrame:
    matrices = {}
    if "age" in growth_info:
        matrices["[row_idx]"] = growth_info["row_idx"]
        matrices["[state.age]"] = growth_info["age"]
    matrices.update(
        {
            "Input.Merch": growth_info["merch_inc"] * 0.5,
            "Input.Other": growth_info["other_inc"] * 0.5,
            "Input.Foliage": growth_info["foliage_inc"] * 0.5,
            "Input.CoarseRoots": growth_info["coarse_root_inc"] * 0.5,
            "Input.FineRoots": growth_info["fine_root_inc"] * 0.5,
        }
    )
    return pd.DataFrame(matrices)


def overmature_decline(
    growth_info: dict[str, np.ndarray],
) -> pd.DataFrame:
    matrices = {}
    if "age" in growth_info:
        matrices["[row_idx]"] = growth_info["row_idx"]
        matrices["[state.age]"] = growth_info["age"]
    matrices.update(
        {
            "Merch.StemSnag": growth_info["merch_to_stem_snag_prop"],
            "Other.BranchSnag": growth_info["other_to_branch_snag_prop"],
            "Other.AboveGroundFastSoil": growth_info["other_to_ag_fast_prop"],
            "Foliage.AboveGroundVeryFastSoil": growth_info[
                "foliage_to_ag_fast_prop"
            ],
            "CoarseRoots.AboveGroundFastSoil": growth_info[
                "coarse_root_to_ag_fast_prop"
            ],
            "CoarseRoots.BelowGroundFastSoil": growth_info[
                "coarse_root_to_bg_fast_prop"
            ],
            "FineRoots.AboveGroundVeryFastSoil": growth_info[
                "fine_root_to_ag_vfast_prop"
            ],
            "FineRoots.BelowGroundVeryFastSoil": growth_info[
                "fine_root_to_bg_vfast_prop"
            ],
        }
    )
    return pd.DataFrame(matrices)


def snag_turnover(
    turnover_params: pd.DataFrame, spinup_format: bool
) -> pd.DataFrame:
    if spinup_format:
        table_name = "parameters"
    else:
        table_name = "state"
    snag_turnover_pool_flows = {
        f"[{table_name}.spatial_unit_id]": turnover_params["spatial_unit_id"],
        f"[{table_name}.sw_hw]": turnover_params["sw_hw"],
        "StemSnag.StemSnag": 1 - turnover_params["StemSnag"],
        "StemSnag.MediumSoil": turnover_params["StemSnag"],
        "BranchSnag.BranchSnag": 1 - turnover_params["BranchSnag"],
        "BranchSnag.AboveGroundFastSoil": turnover_params["BranchSnag"],
    }
    return pd.DataFrame(snag_turnover_pool_flows)


def biomass_turnover(
    turnover_params: pd.DataFrame, spinup_format: bool
) -> pd.DataFrame:
    if spinup_format:
        table_name = "parameters"
    else:
        table_name = "state"
    biomass_turnover_pool_flows = {
        f"[{table_name}.spatial_unit_id]": turnover_params["spatial_unit_id"],
        f"[{table_name}.sw_hw]": turnover_params["sw_hw"],
        "Merch.StemSnag": turnover_params["StemAnnualTurnoverRate"],
        "Foliage.AboveGroundVeryFastSoil": turnover_params["FoliageFallRate"],
        "Other.BranchSnag": turnover_params["OtherToBranchSnagSplit"]
        * turnover_params["BranchTurnoverRate"],
        "Other.AboveGroundFastSoil": (
            (1 - turnover_params["OtherToBranchSnagSplit"])
            * turnover_params["BranchTurnoverRate"]
        ),
        "CoarseRoots.AboveGroundFastSoil": (
            turnover_params["CoarseRootAGSplit"]
            * turnover_params["CoarseRootTurnProp"]
        ),
        "CoarseRoots.BelowGroundFastSoil": (
            (1 - turnover_params["CoarseRootAGSplit"])
            * turnover_params["CoarseRootTurnProp"]
        ),
        "FineRoots.AboveGroundVeryFastSoil": (
            turnover_params["FineRootAGSplit"]
            * turnover_params["FineRootTurnProp"]
        ),
        "FineRoots.BelowGroundVeryFastSoil": (
            1 - turnover_params["FineRootAGSplit"]
        )
        * turnover_params["FineRootTurnProp"],
    }
    return pd.DataFrame(biomass_turnover_pool_flows)


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
) -> pd.DataFrame:
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
    matrix_data = {}
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
        matrix_data[f"{dom_pool}.{dom_pool}"] = 1 - decay_rate
        matrix_data[f"{dom_pool}.{dom_pool_flows[dom_pool]}"] = decay_rate * (
            1 - prop_to_atmosphere
        )
        matrix_data[f"{dom_pool}.CO2"] = decay_rate * prop_to_atmosphere
    decay_ops = pd.DataFrame(matrix_data)
    return decay_ops


def slow_decay(
    mean_annual_temp: np.ndarray, decay_parameters: dict[str, dict[str, float]]
) -> pd.DataFrame:
    matrix_data = {}
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
        matrix_data[f"{dom_pool}.{dom_pool}"] = 1 - decay_rate
        matrix_data[f"{dom_pool}.CO2"] = decay_rate * prop_to_atmosphere

    slow_decay_ops = pd.DataFrame(matrix_data)
    return slow_decay_ops


def slow_mixing(rate: float) -> pd.DataFrame:
    slow_mixing_pool_flows = {
        "AboveGroundSlowSoil.BelowGroundSlowSoil": [rate],
        "AboveGroundSlowSoil.AboveGroundSlowSoil": [1 - rate],
    }
    return pd.DataFrame(slow_mixing_pool_flows)
