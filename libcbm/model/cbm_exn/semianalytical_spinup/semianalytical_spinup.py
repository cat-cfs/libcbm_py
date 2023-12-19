"""
An implementation of CBM-CFS3 carbon dynamics using the semi-analytical
solution[2] to accelerate spin-up.  An adjustment for historical
disturbance return interval is applied to the result[1]

Converts libcbm's iterative matrix aproach to the generalized format
described in Weng et al 2012[3]

[1] Kelly Ann Bona, Cindy Shaw, Dan K. Thompson, Oleksandra Hararuk,
Kara Webster, Gary Zhang, Mihai Voicu, Werner A. Kurz:
The Canadian model for peatlands (CaMP): A peatland carbon model for
national greenhouse gas reporting, Ecological Modelling, Volume 431,
2020, 109164, ISSN 0304-3800,
https://doi.org/10.1016/j.ecolmodel.2020.109164.

[2] Xia, J. Y., Luo, Y. Q., Wang, Y.-P., Weng, E. S., and Hararuk, O.: A
semi-analytical solution to accelerate spin-up of a coupled carbon and
nitrogen land model to steady state, Geosci. Model Dev., 5, 1259–1271,
https://doi.org/10.5194/gmd-5-1259-2012, 2012.

[3] Weng, E., Luo, Y., Wang, W., Wang, H., Hayes, D. J., McGuire, A. D.,
Hastings, A., & Schimel, D. S. (2012). Ecosystem carbon storage capacity
as affected by disturbance regimes: A general theoretical model. Journal
of Geophysical Research: Biogeosciences, 117(G3).
https://doi.org/10.1029/2012JG002040

"""
from typing import Union
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from libcbm import resources
from libcbm.model.model_definition import model_matrix_ops
from libcbm.model.model_definition import matrix_conversion
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn.semianalytical_spinup import (
    semianalytical_spinup_input,
)
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory


class InputMode(Enum):
    """
    use the accumulated biomass at min(maxDefinedAge, return_interval) to
    define steady state input
    """

    MaxDefinedAge = 1

    """
    use the accumulated biomass at min(ageOfPeakBiomass, return_interval) to
    define steady state input
    """
    PeakBiomass = 2

    """
    use the mean input on the interval 0 to return_interval to define steady
    state input
    """
    MeanInput = 3


def get_default_dom_pools() -> list[str]:
    """
    get the list of cbm_exn dead organic matter pool names
    """
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


def get_spinup_matrices(
    spinup_vars: ModelVariables,
    spinup_ops: list[dict],
    pools: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Get the cbm_exn spinup operation matrices dataframes merged onto the
    specified spinup vars

    Args:
        spinup_vars (ModelVariables): the spinup simulation variables, pools
            and state object
        spinup_ops (list[dict]): the unmerged spinup matrix operations
        pools (dict[str, int]): enumeration of pool_name, pool_idx

    Returns:
        dict[str, pd.DataFrame]: the collection of matrices associated with
            each row of spinup vars
    """
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


# TODO this needs a re-work to support bulk iteration
#  def run_iterative(
#      n_steps: int,
#      dom_pools: list[str],
#      Uss: np.ndarray,
#      M: np.ndarray,
#      DM: Union[np.ndarray, None],
#  ) -> pd.DataFrame:
#      results = np.zeros(shape=(n_steps, len(dom_pools)))
#      for t in range(n_steps - 1):
#          if DM is not None:
#              dx = Uss + (results[t, :] @ M) + results[t, :] @ DM
#          else:
#              dx = Uss + (results[t, :] @ M)
#          results[t + 1, :] = results[t, :] + dx
#      return pd.DataFrame(columns=dom_pools, data=results)


def get_step_matrix(
    spinup_matrices: dict[str, pd.DataFrame],
) -> sparse.csc_matrix:
    """Produce a matrix of dom-pool to dom-pool transfers and diagonal dom
    pool C retentions for the CBM dom pool decay and mixing routines.  Each
    row in the specified spinup matrices is assembled into a single
    `scipy.sparse.block_diag`-like formation sparse matrix to perform a bulk
    solve operation.

    Args:
        spinup_matrices (dict[str, pd.DataFrame]): collection of
            dataframe-formatted operation matrices

    Returns:
        sparse.csc_matrix: A sparse matrix containing the dom-pool to dom-pool
            transfers and also the dom pool C retentions along the diagonal.

    """
    dom_pool_dict = {p: i for i, p in enumerate(get_default_dom_pools())}
    spinup_matrices = {
        name: matrix_conversion.filter_pools(dom_pool_dict, mat_df)
        for name, mat_df in spinup_matrices.items()
    }
    coo_mats = {
        k: matrix_conversion.to_coo_matrix(dom_pool_dict, v)
        for k, v in spinup_matrices.items()
    }

    csc_mats = {n: c.tocsc() for n, c in coo_mats.items()}
    spinup_matrix_state_state: sparse.csc_matrix = (
        csc_mats["snag_turnover"]
        @ csc_mats["dom_decay"]
        @ csc_mats["slow_decay"]
        @ csc_mats["slow_mixing"]
    )
    step_matrix = spinup_matrix_state_state - sparse.identity(
        spinup_matrix_state_state.shape[0], format="csc"
    )
    return step_matrix


def get_disturbance_frequency(
    return_interval: np.ndarray,
) -> sparse.dia_matrix:
    """
    Compute the effect of return interval as a frequency to adjust
    disturbance effects for the semianalytical procedure.  This is
    assembled into a block diag formation.

    Args:
        return_interval (np.ndarray): an array of return intervals (1 record
            per stand)

    Returns:
        sparse.dia_matrix: the diagonal tiled disturbance frequency
    """
    n_dom_pools = len(get_default_dom_pools())

    disturbance_frequency = sparse.diags(
        np.repeat(1 / return_interval, n_dom_pools)
    )
    return disturbance_frequency


def get_disturbance_matrix(
    spinup_matrices: dict[str, pd.DataFrame],
) -> sparse.csc_matrix:
    """Create a block_diag sparse matrix of the disturbance effects
    on dom:
        * The diagonal is the losses to atmosphere and the
        * off-diagonals are the dom-pool to dom-pool transfers.

    Each row of the specified spinup matrices is assembled into a
    block-diagonal CSC formatted sparse matrix.

    Args:
        spinup_matrices (dict[str, pd.DataFrame]): The collection of
            spinup matrices in dataframe form

    Returns:
        sparse.csc_matrix: The disturbance effects on dom as a sparse block
            diagonal CSC matrix
    """
    dom_pool_dict = {p: i for i, p in enumerate(get_default_dom_pools())}
    disturbance_mat_dom = matrix_conversion.filter_pools(
        dom_pool_dict, spinup_matrices["disturbance"]
    )
    disturbance_mat_coo = matrix_conversion.to_coo_matrix(
        dom_pool_dict, disturbance_mat_dom
    )

    disturbance_mat_csc = disturbance_mat_coo.tocsc()
    identity = sparse.identity(disturbance_mat_csc.shape[0], format="csc")
    M_dm = disturbance_mat_csc - identity
    return M_dm


def semianalytical_spinup(
    spinup_input: dict[str, pd.DataFrame],
    input_mode: InputMode,
    parameters: CBMEXNParameters
) -> pd.DataFrame:
    """
    Use the semi-analytical approach for spinup parameterized by the CBM-CFS3
    approach to estimate end-of-spinup DOM C pools.

    Args:
        spinup_input (dict[str, pd.DataFrame]): spinup input
        input_mode (InputMode): on of the input modes for estimating
            the steady state inputs to the dead organic matter pools.

    Raises:
        NotImplementedError: Raise on a not-yet-implemented input_mode

    Returns:
        pd.DataFrame: A dataframe containing 1 row for each stand and 1
            column for each cbm_exn dead organic matter pool.  The dataframe's
            values are the estimate for end-of-spinup dead oranic matter for
            each stand.
    """

    n_rows = len(spinup_input["parameters"].index)

    pool_dict = {p: i for i, p in enumerate(parameters.pool_configuration())}
    dom_pools = get_default_dom_pools()

    spinup_vars = cbm_exn_spinup.prepare_spinup_vars(
        ModelVariables.from_pandas(spinup_input),
        parameters,
    )
    spinup_vars["state"]["disturbance_type"].assign(
        spinup_vars["parameters"]["historical_disturbance_type"]
    )
    spinup_ops = cbm_exn_spinup.get_default_ops(parameters, spinup_vars)
    spinup_matrices = get_spinup_matrices(spinup_vars, spinup_ops, pool_dict)
    return_interval = spinup_vars["parameters"]["return_interval"].to_numpy()
    if input_mode == InputMode.MaxDefinedAge:
        bio = semianalytical_spinup_input.get_bio_at_max_age(
            spinup_ops, spinup_vars
        )
    elif input_mode == InputMode.PeakBiomass:
        bio = semianalytical_spinup_input.get_bio_at_peak(
            spinup_ops, spinup_vars
        )
    else:
        raise NotImplementedError(
            f"specified input_mode: {input_mode} not yet implemented"
        )
    Uss = (
        semianalytical_spinup_input.get_steady_state_input(
            pool_dict,
            dom_pools,
            bio,
            spinup_matrices,
        )
        .to_numpy()
        .flatten()
    )

    M = get_step_matrix(spinup_matrices)
    DM = get_disturbance_matrix(spinup_matrices)
    f = get_disturbance_frequency(return_interval)
    result: np.ndarray = -linalg.spsolve((M.T + (DM @ f).T), Uss)
    return pd.DataFrame(
        columns=dom_pools, data=result.reshape(n_rows, len(dom_pools))
    )


def create_spinup_seed(
    semi_analytical_result: pd.DataFrame,
    spinup_input: dict[str, pd.DataFrame],
    parameters: CBMEXNParameters,
) -> dict[str, pd.DataFrame]:

    spinup_seed_pools = pd.DataFrame(
        {
            p: semi_analytical_result[p]
            if p in semi_analytical_result.columns else 0.0
            for p in parameters.pool_configuration()
        }
    )
    spinup_seed_pools["Input"] = 1.0
    seed_spinup_input = {
        k: v.copy() if k == "parameters" else v
        for k, v in spinup_input.items()
    }
    seed_spinup_input["parameters"]["min_rotations"] = 1
    seed_spinup_input["parameters"]["max_rotations"] = 2
    seed_spinup_input["pools"] = spinup_seed_pools
    return seed_spinup_input


def prepare_parameters(
    parameters: Union[dict, None] = None,
    config_path: Union[str, None] = None,
) -> CBMEXNParameters:

    if not config_path:
        config_path = resources.get_cbm_exn_parameters_dir()

    params = parameters_factory(config_path, parameters)
    return params
