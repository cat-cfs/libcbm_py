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
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from libcbm.model.cbm_exn.semianalytical_spinup import matrix_operations
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory
from libcbm import resources
from libcbm.model.cbm_exn import cbm_exn_spinup


def run_iterative(
    n_steps: int,
    dom_pools: list[str],
    Uss: np.ndarray,
    M: np.ndarray,
    DM: Union[np.ndarray, None],
) -> pd.DataFrame:
    results = np.zeros(shape=(n_steps, len(dom_pools)))
    for t in range(n_steps - 1):
        if DM is not None:
            dx = Uss + (results[t, :] @ M) + results[t, :] @ DM
        else:
            dx = Uss + (results[t, :] @ M)
        results[t + 1, :] = results[t, :] + dx
    return pd.DataFrame(columns=dom_pools, data=results)


def get_step_matrix(
    spinup_matrices: dict[str, pd.DataFrame],
) -> sparse.csc_matrix:
    dom_pool_dict = {
        p: i for i, p in enumerate(matrix_operations.get_default_dom_pools())
    }
    spinup_matrices = {
        name: matrix_operations.filter_pools(dom_pool_dict, mat_df)
        for name, mat_df in spinup_matrices.items()
    }
    coo_mats = {
        k: matrix_operations.to_coo_matrix(dom_pool_dict, v)
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
    n_dom_pools = len(matrix_operations.get_default_dom_pools())

    disturbance_frequency = sparse.diags(
        np.tile(1 / return_interval, n_dom_pools)
    )
    return disturbance_frequency


def get_disturbance_matrix(
    spinup_matrices: dict[str, pd.DataFrame],
) -> sparse.csc_matrix:
    dom_pool_dict = {
        p: i for i, p in enumerate(matrix_operations.get_default_dom_pools())
    }
    disturbance_mat_dom = matrix_operations.filter_pools(
        dom_pool_dict, spinup_matrices["disturbance"]
    )
    disturbance_mat_coo = matrix_operations.to_coo_matrix(
        dom_pool_dict, disturbance_mat_dom
    )

    disturbance_mat_csc = disturbance_mat_coo.tocsc()
    identity = sparse.identity(disturbance_mat_csc.shape[0], format="csc")
    M_dm = disturbance_mat_csc - identity
    return M_dm


def get_steady_state_input(
    pool_dict: dict[str, int],
    dom_pools: list[str],
    steady_state_bio: pd.DataFrame,
    spinup_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    bio_turover_df = spinup_matrices["biomass_turnover"]
    bio_turnover_mat = matrix_operations.to_coo_matrix(
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


def semianalytical_spinup(
    spinup_input: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    n_rows = len(spinup_input["parameters"].index)
    default_parameters = parameters_factory(
        resources.get_cbm_exn_parameters_dir()
    )
    pool_dict = {
        p: i for i, p in enumerate(default_parameters.pool_configuration())
    }
    dom_pools = matrix_operations.get_default_dom_pools()

    spinup_vars = cbm_exn_spinup.prepare_spinup_vars(
        ModelVariables.from_pandas(spinup_input),
        default_parameters,
    )
    spinup_vars["state"]["disturbance_type"].assign(
        spinup_vars["parameters"]["last_pass_disturbance_type"]
    )
    spinup_ops, _ = matrix_operations.get_default_spinup_ops(spinup_vars)
    spinup_matrices = matrix_operations.get_spinup_matrices(
        spinup_vars, spinup_ops, pool_dict
    )
    return_interval = spinup_vars["parameters"]["return_interval"].to_numpy()
    Uss = (
        get_steady_state_input(
            pool_dict,
            dom_pools,
            matrix_operations.get_bio_at_max_age(spinup_ops, spinup_vars),
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
