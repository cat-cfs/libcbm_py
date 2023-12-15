import pandas as pd
import numpy as np
from scipy import sparse
from libcbm import resources
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory
from libcbm.model.model_definition import model_matrix_ops


def filter_pools(
    pool_idx: dict[str, int],
    op_data: pd.DataFrame,
) -> pd.DataFrame:
    included_col_idx = list()
    for i_col, col in enumerate(op_data.columns):
        source, sink = col.split(".")
        if source in pool_idx and sink in pool_idx:
            included_col_idx.append(i_col)
    return op_data.iloc[:, included_col_idx].copy()


def to_coo_matrix(
    pool_idx: dict[str, int],
    op_data: pd.DataFrame,
) -> sparse.coo_matrix:
    n_pools = len(pool_idx)
    n_rows = len(op_data.index)
    diag_coords = set(zip(range(n_pools), range(n_pools)))
    nonzero_coords = {}
    for col in op_data.columns:
        source, sink = col.split(".")
        row_idx = pool_idx[source]
        col_idx = pool_idx[sink]
        nonzero_coords[(row_idx, col_idx)] = op_data[col].to_numpy()

    all_coords = diag_coords.union(nonzero_coords.keys())

    n_coords = len(all_coords)

    out_rows = np.zeros(n_coords * n_rows, dtype="int")
    out_cols = np.zeros(n_coords * n_rows, dtype="int")
    out_values = np.zeros(n_coords * n_rows, dtype="float")

    for idx, (row_idx, col_idx) in enumerate(all_coords):
        if (row_idx, col_idx) in nonzero_coords:
            value = nonzero_coords[(row_idx, col_idx)]
        else:
            value = 1.0
        out_rows[idx::n_coords] = row_idx + np.arange(
            0, n_pools * n_rows, n_pools
        )
        out_cols[idx::n_coords] = col_idx + np.arange(
            0, n_pools * n_rows, n_pools
        )
        out_values[idx::n_coords] = value
    return sparse.coo_matrix((out_values, (out_rows, out_cols)))


def to_coo_matrices(
    pool_idx: dict[str, int],
    op_data: dict[str, pd.DataFrame],
) -> dict[str, sparse.coo_matrix]:
    return {k: to_coo_matrix(pool_idx, v) for k, v in op_data.items()}


def get_default_pools() -> list[str]:
    params = parameters_factory(resources.get_cbm_exn_parameters_dir())
    return params.pool_configuration()


def get_default_dom_pools():
    return [
        "AboveGroundVeryFastSoil",
        "BelowGroundVeryFastSoil",
        "AboveGroundFastSoil",
        "BelowGroundFastSoil",
        "MediumSoil",
        "AboveGroundSlowSoil",
        "BelowGroundSlowSoil",
        "StemSnag",
        "BranchSnag",
    ]


def get_default_spinup_ops(spinup_vars: ModelVariables) -> tuple[list, list]:
    params = parameters_factory(resources.get_cbm_exn_parameters_dir())
    spinup_op_list = cbm_exn_spinup.get_default_op_list()
    spinup_ops = cbm_exn_spinup.get_default_ops(params, spinup_vars)
    return spinup_ops, spinup_op_list


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


def get_spinup_matrices(
    spinup_vars: ModelVariables,
    spinup_ops: list[dict],
    pools: dict[str, int],
) -> dict[str, pd.DataFrame]:
    pool_names = set(pools.keys())
    output: dict[str, pd.DataFrame] = {}
    for o in spinup_ops:
        op_dataframe = model_matrix_ops.prepare_operation_dataframe(
            o["op_data"], pool_names
        )
        matrix_index = model_matrix_ops.init_index(op_dataframe)
        idx = matrix_index.compute_matrix_index(
            spinup_vars,
            default_matrix_index=(
                o["default_matrix_index"]
                if "default_matrix_index" in o
                else None
            ),
        )
        output[o["name"]] = op_dataframe.iloc[idx]

    return output
