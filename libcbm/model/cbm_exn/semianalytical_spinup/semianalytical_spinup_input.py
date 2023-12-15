import pandas as pd
import numpy as np

from libcbm.model.model_definition import matrix_conversion
from libcbm.model.model_definition.model_variables import ModelVariables


def get_cumulative_growth(spinup_ops: list[dict]):
    all_spinup_ops_by_name: dict[str, pd.DataFrame] = {
        o["name"]: o["op_data"] for o in spinup_ops
    }
    growth_ops = all_spinup_ops_by_name["growth"]
    growth_ops_grouped = (
        growth_ops.sort_values(by=["[row_idx]", "[state.age]"])
        .groupby(["[row_idx]"])[
            [
                "Input.Merch",
                "Input.Foliage",
                "Input.Other",
                "Input.CoarseRoots",
                "Input.FineRoots",
            ]
        ]
        .cumsum()
        .rename(
            columns={
                "Input.Merch": "Merch",
                "Input.Foliage": "Foliage",
                "Input.Other": "Other",
                "Input.CoarseRoots": "CoarseRoots",
                "Input.FineRoots": "FineRoots",
            }
        )
        * 2.0
    )
    growth_ops_grouped.insert(0, "[row_idx]", growth_ops["[row_idx]"])
    growth_ops_grouped.insert(1, "[state.age]", growth_ops["[state.age]"])
    return growth_ops_grouped


def get_bio_at_max_age(
    spinup_ops: list[dict], spinup_vars: ModelVariables
) -> pd.DataFrame:
    """Get the maximum biomass in the range of
    age=0 to age=min(max(age), return_interval)

    Args:
        spinup_ops (list[dict]): _description_
        spinup_vars (ModelVariables): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cumulative_growth = get_cumulative_growth(spinup_ops)
    max_ages = cumulative_growth.groupby(["[row_idx]"])["[state.age]"].max()
    return_intervals = spinup_vars["parameters"].to_pandas()["return_interval"]
    max_age_return_interval = pd.DataFrame(
        data={"max_age": max_ages, "return_interval": return_intervals}
    )
    max_age_return_interval["peak_spinup_age"] = max_age_return_interval.min(
        axis=1
    )
    max_age_return_interval_merge = max_age_return_interval.reset_index()
    bio_max_age = max_age_return_interval_merge.merge(
        cumulative_growth,
        left_on=["index", "peak_spinup_age"],
        right_on=["[row_idx]", "[state.age]"],
        how="left",
    )
    bio_max_age = bio_max_age[cumulative_growth.columns]
    bio_max_age = bio_max_age.set_index(["[row_idx]", "[state.age]"])
    return bio_max_age


def get_bio_at_peak(
    spinup_ops: list[dict], spinup_vars: ModelVariables
) -> pd.DataFrame:
    cumulative_growth = get_cumulative_growth(spinup_ops)
    bio_max_bio = cumulative_growth.copy()
    max_ages = cumulative_growth.groupby(["[row_idx]"])["[state.age]"].max()
    return_intervals = spinup_vars["parameters"].to_pandas()["return_interval"]
    max_age_return_interval = pd.DataFrame(
        data={"max_age": max_ages, "return_interval": return_intervals}
    )
    max_age_return_interval["peak_spinup_age"] = max_age_return_interval.min(
        axis=1
    )
    bio_max_bio = bio_max_bio.merge(
        max_age_return_interval[["peak_spinup_age"]],
        left_on="[row_idx]",
        right_index=True,
        how="left",
    )
    bio_max_bio = bio_max_bio[
        bio_max_bio["[state.age]"] <= bio_max_bio["peak_spinup_age"]
    ]
    bio_max_bio["total"] = bio_max_bio[
        ["Merch", "Foliage", "Other", "CoarseRoots", "FineRoots"]
    ].sum(axis=1)
    bio_max_bio = (
        bio_max_bio.sort_values(by="total", ascending=False)
        .drop_duplicates("[row_idx]")
        .sort_values(by=["[row_idx]", "[state.age]"])
    )
    bio_max_bio = bio_max_bio[cumulative_growth.columns]
    bio_max_bio = bio_max_bio.set_index(["[row_idx]", "[state.age]"])
    return bio_max_bio


def get_steady_state_input(
    pool_dict: dict[str, int],
    dom_pools: list[str],
    steady_state_bio: pd.DataFrame,
    spinup_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    bio_turover_df = spinup_matrices["biomass_turnover"]
    bio_turnover_mat = matrix_conversion.to_coo_matrix(
        pool_dict, bio_turover_df
    ).tocsc()
    n_rows = len(bio_turover_df.index)
    pools = np.zeros(shape=(n_rows, len(pool_dict)))

    pools[
        :, slice(pool_dict["Merch"], pool_dict["FineRoots"] + 1)
    ] = steady_state_bio.to_numpy()
    input = pools.flatten() @ bio_turnover_mat
    return pd.DataFrame(
        columns=list(pool_dict.keys()),
        data=input.reshape((n_rows, len(pool_dict))),
    )[dom_pools]
