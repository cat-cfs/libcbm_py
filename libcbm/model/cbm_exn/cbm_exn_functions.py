import pandas as pd
import numpy as np
import numba as nb
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


def total_root_bio_hw(
    root_parameters: dict[str, float], total_ag_bio_t: np.ndarray
):
    return root_parameters["rb_hw_a"] * np.power(
        total_ag_bio_t / root_parameters["biomass_to_carbon_rate"],
        root_parameters["rb_hw_b"],
    )


def total_root_bio_sw(
    root_parameters: dict[str, float], total_ag_bio_t: np.ndarray
) -> np.ndarray:
    return (
        root_parameters["rb_sw_a"]
        * total_ag_bio_t
        / root_parameters["biomass_to_carbon_rate"]
    )


def fine_root_proportion(
    root_parameters: dict[str, float], total_root_bio
) -> np.ndarray:
    return root_parameters["frp_a"] + root_parameters["frp_b"] * np.exp(
        root_parameters["frp_c"] * total_root_bio
    )


def compute_root_inc(
    species_id: np.ndarray,
    merch: np.ndarray,
    foliage: np.ndarray,
    other: np.ndarray,
    coarse_root: np.ndarray,
    fine_root: np.ndarray,
    merch_inc: np.ndarray,
    foliage_inc: np.ndarray,
    other_inc: np.ndarray,
    parameters: CBMEXNParameters,
) -> dict[str, np.ndarray]:
    total_ag_bio_t = (
        merch + merch_inc + foliage + foliage_inc + other + other_inc
    )

    # sw=0, hw=1
    species_id_series = pd.Series(species_id)
    sw_hw_map = parameters.get_sw_hw_map()
    missing_species = set(species_id_series.unique()).difference(
        set(sw_hw_map.keys())
    )
    if missing_species:
        raise ValueError(
            "the following species ids found in state.species"
            f"array are not defined in default paramters {missing_species}"
        )
    sw_hw = species_id_series.map(sw_hw_map)

    root_parameters = parameters.get_root_parameters()
    total_root_bio = np.where(
        sw_hw,
        total_root_bio_sw(root_parameters, total_ag_bio_t),
        total_root_bio_hw(root_parameters, total_ag_bio_t),
    )
    fine_root_prop = fine_root_proportion(root_parameters, total_root_bio)
    coarse_root_inc = (
        total_root_bio
        * (1 - fine_root_prop)
        * root_parameters["biomass_to_carbon_rate"]
        - coarse_root
    )
    fine_root_inc = (
        total_root_bio
        * fine_root_prop
        * root_parameters["biomass_to_carbon_rate"]
        - fine_root
    )
    return {"coarse_root_inc": coarse_root_inc, "fine_root_inc": fine_root_inc}


@nb.njit()
def _overmature_decline_compute(
    merch: np.ndarray,
    foliage: np.ndarray,
    other: np.ndarray,
    coarse_root: np.ndarray,
    fine_root: np.ndarray,
    merch_inc: np.ndarray,
    foliage_inc: np.ndarray,
    other_inc: np.ndarray,
    coarse_root_inc: np.ndarray,
    fine_root_inc: np.ndarray,
    other_to_branch_snag_split: np.ndarray,
    coarse_root_ag_split: np.ndarray,
    fine_root_ag_split: np.ndarray,
    merch_to_stem_snag_prop: np.ndarray,
    other_to_branch_snag_prop: np.ndarray,
    other_to_ag_fast_prop: np.ndarray,
    foliage_to_ag_fast_prop: np.ndarray,
    coarse_root_to_ag_fast_prop: np.ndarray,
    coarse_root_to_bg_fast_prop: np.ndarray,
    fine_root_to_ag_vfast_prop: np.ndarray,
    fine_root_to_bg_vfast_prop: np.ndarray,
):

    tolerance = -0.0001
    size = merch.shape[0]
    for i in range(size):
        overmature = (
            merch_inc[i]
            + foliage_inc[i]
            + other_inc[i]
            + fine_root_inc[i]
            + coarse_root_inc[i]
        ) < tolerance
        if overmature and merch_inc[i] < 0:
            merch_to_stem_snag_prop[i] = -merch_inc[i] / merch[i]
        if overmature and other_inc[i] < 0:
            other_to_branch_snag_prop[i] = (
                -other_inc[i] * other_to_branch_snag_split[i] / other[i]
            )
            other_to_ag_fast_prop[i] = (
                -other_inc[i] * (1 - other_to_branch_snag_split[i]) / other[i]
            )
        if overmature and foliage_inc[i] < 0:
            foliage_to_ag_fast_prop[i] = -foliage_inc[i] / foliage[i]
        if overmature and coarse_root_inc[i] < 0:
            coarse_root_to_ag_fast_prop[i] = (
                -coarse_root_inc[i] * coarse_root_ag_split[i] / coarse_root[i]
            )
            coarse_root_to_bg_fast_prop[i] = (
                -coarse_root_inc[i]
                * (1 - coarse_root_ag_split[i])
                / coarse_root[i]
            )
        if overmature and fine_root_inc[i] < 0:
            fine_root_to_ag_vfast_prop[i] = (
                -fine_root_inc[i] * fine_root_ag_split[i] / fine_root[i]
            )
            fine_root_to_bg_vfast_prop[i] = (
                -fine_root_inc[i] * (1 - fine_root_ag_split[i]) / fine_root[i]
            )


def compute_overmature_decline(
    spatial_unit_id: np.ndarray,
    merch: np.ndarray,
    foliage: np.ndarray,
    other: np.ndarray,
    coarse_root: np.ndarray,
    fine_root: np.ndarray,
    merch_inc: np.ndarray,
    foliage_inc: np.ndarray,
    other_inc: np.ndarray,
    coarse_root_inc: np.ndarray,
    fine_root_inc: np.ndarray,
    parameters: CBMEXNParameters,
) -> dict[str, np.ndarray]:
    turnover_parameters = parameters.get_turnover_parameters()[
        [
            "spatial_unit_id",
            "OtherToBranchSnagSplit",
            "CoarseRootAGSplit",
            "FineRootAGSplit",
        ]
    ].merge(
        pd.Series(
            name="spatial_unit_id",
            data=spatial_unit_id,
        ),
        left_on="spatial_unit_id",
        right_on="spatial_unit_id",
    )

    if len(turnover_parameters.index) != spatial_unit_id.shape[0]:
        raise ValueError()

    other_to_branch_snag_split = turnover_parameters[
        "OtherToBranchSnagSplit"
    ].to_numpy()
    coarse_root_ag_split = turnover_parameters["CoarseRootAGSplit"].to_numpy()
    fine_root_ag_split = turnover_parameters["FineRootAGSplit"].to_numpy()
    merch_to_stem_snag_prop = np.zeros(spatial_unit_id.shape)
    other_to_branch_snag_prop = np.zeros(spatial_unit_id.shape)
    other_to_ag_fast_prop = np.zeros(spatial_unit_id.shape)
    foliage_to_ag_fast_prop = np.zeros(spatial_unit_id.shape)
    coarse_root_to_ag_fast_prop = np.zeros(spatial_unit_id.shape)
    coarse_root_to_bg_fast_prop = np.zeros(spatial_unit_id.shape)
    fine_root_to_ag_vfast_prop = np.zeros(spatial_unit_id.shape)
    fine_root_to_bg_vfast_pro = np.zeros(spatial_unit_id.shape)
    _overmature_decline_compute(
        merch,
        foliage,
        other,
        coarse_root,
        fine_root,
        merch_inc,
        foliage_inc,
        other_inc,
        coarse_root_inc,
        fine_root_inc,
        other_to_branch_snag_split,
        coarse_root_ag_split,
        fine_root_ag_split,
        merch_to_stem_snag_prop,
        other_to_branch_snag_prop,
        other_to_ag_fast_prop,
        foliage_to_ag_fast_prop,
        coarse_root_to_ag_fast_prop,
        coarse_root_to_bg_fast_prop,
        fine_root_to_ag_vfast_prop,
        fine_root_to_bg_vfast_pro,
    )
    return {
        "merch_to_stem_snag_prop": merch_to_stem_snag_prop,
        "other_to_branch_snag_prop": other_to_branch_snag_prop,
        "other_to_ag_fast_prop": other_to_ag_fast_prop,
        "foliage_to_ag_fast_prop": foliage_to_ag_fast_prop,
        "coarse_root_to_ag_fast_prop": coarse_root_to_ag_fast_prop,
        "coarse_root_to_bg_fast_prop": coarse_root_to_bg_fast_prop,
        "fine_root_to_ag_vfast_prop": fine_root_to_ag_vfast_prop,
        "fine_root_to_bg_vfast_prop": fine_root_to_bg_vfast_pro,
    }


def prepare_growth_info(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> dict[str, np.ndarray]:

    merch = cbm_vars["pools"]["Merch"].to_numpy()
    foliage = cbm_vars["pools"]["Foliage"].to_numpy()
    other = cbm_vars["pools"]["Other"].to_numpy()
    fine_root = cbm_vars["pools"]["FineRoot"].to_numpy()
    coarse_root = cbm_vars["pools"]["CoarseRoots"].to_numpy()
    species_id = cbm_vars["state"]["species"].to_numpy()
    spatial_unit_id = cbm_vars["state"]["spatial_unit_id"].to_numpy()
    merch_inc = np.maximum(
        merch,
        -(cbm_vars["parameters"]["merch_inc"].to_numpy()),
    )
    foliage_inc = np.maximum(
        foliage,
        -(cbm_vars["parameters"]["foliage_inc"].to_numpy()),
    )
    other_inc = np.maximum(
        other,
        -(cbm_vars["parameters"]["other_inc"].to_numpy()),
    )

    root_inc = compute_root_inc(
        species_id,
        merch,
        foliage,
        other,
        coarse_root,
        fine_root,
        merch_inc,
        foliage_inc,
        other_inc,
        parameters,
    )
    overmature_decline = compute_overmature_decline(
        spatial_unit_id,
        merch,
        foliage,
        other,
        coarse_root,
        fine_root,
        merch_inc,
        foliage_inc,
        other_inc,
        root_inc["coarse_root_inc"],
        root_inc["fine_root_inc"],
        parameters,
    )

    data = {
        "merch_inc": merch_inc,
        "other_inc": other_inc,
        "foliage_inc": foliage_inc,
    }

    data.update(root_inc)
    data.update(overmature_decline)

    return data
