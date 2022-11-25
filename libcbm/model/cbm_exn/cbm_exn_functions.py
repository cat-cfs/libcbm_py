from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


def prepare_growth_info(
    cbm_vars: CBMVariables, parameters: CBMEXNParameters, is_spinup: bool
) -> DataFrame:

    data = {
        "matrix_idx": None,
        "merch_inc": None,
        "other_inc": None,
        "foliage_inc": None,
        "coarse_root_inc": None,
        "fine_root_inc": None,
        "merch_to_stem_snag_prop": None,
        "other_to_branch_snag_prop": None,
        "other_to_ag_fast_prop": None,
        "foliage_to_ag_fast_prop": None,
        "coarse_root_to_bg_fast_prop": None,
        "fine_root_to_ag_vfast_prop": None,
        "fine_root_to_bg_vfast_prop": None,
    }
