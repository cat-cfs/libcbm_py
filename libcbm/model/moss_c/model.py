# moss c model
#
# publication:
# Bona, Kelly & Shaw, Cindy & Fyles, James & Kurz, Werner. (2016).
# Modelling moss-derived carbon in upland black spruce forests.
# Canadian Journal of Forest Research. 46. 10.1139/cjfr-2015-0512.
#
from enum import IntEnum
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

import numba
import numba.typed

from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources


class SpinupState(IntEnum):
    AnnualProcesses = 1,
    HistoricalEvent = 2,
    LastPassEvent = 3,
    End = 4


class Pool(IntEnum):
    Input = 0,
    FeatherMossLive = 1,
    SphagnumMossLive = 2,
    FeatherMossFast = 3,
    SphagnumMossFast = 4,
    FeatherMossSlow = 5,
    SphagnumMossSlow = 6,
    CO2 = 7


def f1(merch_vol, a, b):
    """Returns Canopy openess, O(t) as a function of Merch. Volume

    10^(((a)*(Log(V(t))) + b)

    Args:
        merch_vol (float, np.ndarray): array of merch volumes
        a (float, np.ndarray): MossC a parameter
        b (float, np.ndarray): MossC b parameter

    Returns:
        np.ndarray: Canopy openess
    """
    result = np.full_like(merch_vol, 60.0)
    result[merch_vol != 0.0] = np.power(
        10, a * np.log10(merch_vol[merch_vol != 0.0]) + b)
    return result


def f2(openness, stand_age, c, d):
    """ returns Feather moss ground cover, GCfm(t)

    (O(t)* c) + d

    Args:
        openness (float, np.ndarray): canopy openness
        stand_age (int, np.ndarray): stand age in years
        c (float, np.ndarray): MossC c parameter,
        d (float, np.ndarray): MossC d parameter

    Returns:
        np.ndarray: Feather moss ground cover
    """
    return np.where(
        stand_age < 10,
        0.0,
        np.where(
            openness > 70.0,
            100.0, openness * c + d))


def f3(openness, stand_age, e, f):
    """ returns Sphagnum moss ground cover, GCsp(t)

    (O(t)* e) + f

    Args:
        openness (float, np.ndarray): canopy openness
        stand_age (int, np.ndarray): stand age in years
        e (float, np.ndarray): MossC e parameter,
        f (float, np.ndarray): MossC f parameter

    Returns:
        np.ndarray: Sphagnum moss ground cover
    """
    return np.where(
        stand_age < 10,
        0.0,
        np.where(
            openness > 70.0,
            100.0, openness * e + f))


def f4(openness, g, h):
    """Feathermoss NPP (assuming 100% ground cover), NPPfm

    g*O(t)^h

    Args:
        openness (float, np.ndarray): canopy openness
        g (float, np.ndarray): MossC g parameter,
        h (float, np.ndarray): MossC h parameter,
    """
    return np.where(
        openness < 5.0, 0.6,
        g * np.power(openness, h))


def f5(openness, i, j, l):
    """Sphagnum NPP (assuming 100% ground cover), NPPsp

    i*(O(t)^2) + j*(O(t) + l

    Args:
        openness (float, np.ndarray): canopy openness
        i (float, np.ndarray): MossC i parameter,
        j (float, np.ndarray): MossC j parameter,
        l (float, np.ndarray): MossC l parameter,

    Returns:
        np.ndarray: Sphagnum moss NPP
    """
    return i * openness ** 2.0 + j * openness + l


def f6(merch_vol, m, n):
    """

    Args:
        merch_vol (float, np.ndarray): merchantable volume.
        m (float, np.ndarray): MossC m parameter,
        n (float, np.ndarray): MossC n parameter,
    """
    return np.log(merch_vol) * m + n


def f7(mean_annual_temp, base_decay_rate, q10, t_ref):
    """Applied Decay rate, ak this applies to any of the moss DOM
    pools feather moss fast (kff), feather moss slow (kfs), sphagnum
    fast (ksf), and sphagnum slow (kss)

    Args:
        mean_annual_temp (float, np.ndarray): [description]
        base_decay_rate (float, np.ndarray): [description]
        q10 (float, np.ndarray): [description]
        t_ref (float, np.ndarray): [description]
    """
    return base_decay_rate * np.exp(
        (mean_annual_temp - t_ref) * np.log(q10) * 0.1)


def annual_process_dynamics(state, params):

    kss = f6(params.max_merch_vol, params.m, params.n)
    openness = f1(state.merch_vol, params.a, params.b)

    return SimpleNamespace(
        kss=kss,
        openness=openness,

        # applied feather moss fast pool decay rate
        akff=f7(
            params.mean_annual_temp, params.kff, params.q10,
            params.tref),

        # applied feather moss slow pool decay rate
        akfs=f7(
            params.mean_annual_temp, params.kfs, params.q10,
            params.tref),

        # applied sphagnum fast pool applied decay rate
        aksf=f7(
            params.mean_annual_temp, params.ksf, params.q10,
            params.tref),

        # applied sphagnum slow pool applied decay rate
        akss=f7(
            params.mean_annual_temp, kss, params.q10, params.tref),

        # Feather moss ground cover
        GCfm=f2(openness, state.age, params.c, params.d),

        # Sphagnum ground cover
        GCsp=f3(openness, state.age, params.e, params.f),

        # Feathermoss NPP (assuming 100% ground cover)
        NPPfm=f4(openness, params.g, params.h),

        # Sphagnum NPP (assuming 100% ground cover)
        NPPsp=f5(
            openness, params.i, params.j, params.l))


def expand_matrix(mat, initialize_identity=True):
    n_mats = len(mat[0][2])
    n_rows = len(mat)
    out_rows = [float(mat[r][0]) for r in range(0, n_rows)]
    out_cols = [float(mat[r][1]) for r in range(0, n_rows)]
    out_values = [
        np.array([mat[r][2]])
        if np.isscalar(mat[r][2])
        else np.array(mat[r][2])
        for r in range(0, n_rows)]
    if initialize_identity:
        identity_set = {int(p) for p in Pool}

        for r in range(0, n_rows):
            if out_rows[r] == out_cols[r]:
                identity_set.remove(int(out_rows[r]))
    else:
        identity_set = []
    return __expand_matrix(
        n_mats, n_rows,
        numba.typed.List(out_rows),
        numba.typed.List(out_cols),
        numba.typed.List(out_values),
        numba.typed.List(identity_set))


@numba.njit
def __expand_matrix(n_mats, n_rows, out_rows, out_cols, out_values,
                    identity_set):

    n_output_rows = n_rows + len(identity_set)
    output = [np.zeros(shape=(n_output_rows, 3)) for _ in range(0, n_mats)]
    for i in range(0, n_mats):
        for r, pool in enumerate(identity_set):
            output[i][r][0] = pool
            output[i][r][1] = pool
            output[i][r][2] = 1.0
    for i in range(0, n_mats):
        for r in range(0, n_rows):
            r_offset = r + len(identity_set)
            output[i][r_offset][0] = out_rows[r]
            output[i][r_offset][1] = out_cols[r]
            if out_values[r].size == 1:
                output[i][r_offset][2] = out_values[r][0]
            else:
                output[i][r_offset][2] = out_values[r][i]

    return output


def get_annual_process_matrix(dynamics_param):
    mat = [
        [Pool.Input, Pool.FeatherMossLive,
         dynamics_param.NPPfm * dynamics_param.GCfm / 100.0],
        [Pool.Input, Pool.SphagnumMossLive,
         dynamics_param.NPPsp * dynamics_param.GCsp / 100.0],

        # turnovers
        [Pool.FeatherMossLive, Pool.FeatherMossFast, 1.0],
        [Pool.FeatherMossLive, Pool.FeatherMossLive, 0.0],

        [Pool.FeatherMossFast, Pool.FeatherMossSlow,
         dynamics_param.akff * 0.15],

        [Pool.SphagnumMossLive, Pool.SphagnumMossFast, 1.0],
        [Pool.SphagnumMossLive, Pool.SphagnumMossLive, 0.0],

        [Pool.SphagnumMossFast, Pool.SphagnumMossSlow,
         dynamics_param.aksf * 0.15],

        # fast losses
        [Pool.FeatherMossFast, Pool.FeatherMossFast,
         1.0 - dynamics_param.akff],
        [Pool.SphagnumMossFast, Pool.SphagnumMossFast,
         1.0 - dynamics_param.aksf],

        # decays
        [Pool.FeatherMossFast, Pool.CO2, dynamics_param.akff * 0.85],

        [Pool.SphagnumMossFast, Pool.CO2, dynamics_param.aksf * 0.85],

        [Pool.FeatherMossSlow, Pool.CO2, dynamics_param.akfs],
        [Pool.FeatherMossSlow, Pool.FeatherMossSlow,
         1.0 - dynamics_param.akfs],

        [Pool.SphagnumMossSlow, Pool.CO2, dynamics_param.akss],
        [Pool.SphagnumMossSlow, Pool.SphagnumMossSlow,
         1.0 - dynamics_param.akss]
    ]
    return mat


def get_disturbance_flows(disturbance_type_name, disturbance_matrices):
    matrix = [
        [

        ]
    ]
    return matrix


def _small_slow_diff(last_rotation_slow, this_rotation_slow):
    return abs((last_rotation_slow - this_rotation_slow)
               / (last_rotation_slow+this_rotation_slow)/2.0) < 0.001


def advance_spinup_state(spinup_state, age, return_interval, rotation_num,
                         max_rotations, last_rotation_slow,
                         this_rotation_slow):
    np.where(
        spinup_state == SpinupState.AnnualProcesses,
        np.where(
            age >= return_interval,
            np.where(
                _small_slow_diff(last_rotation_slow, this_rotation_slow)
                | rotation_num > max_rotations,
                SpinupState.LastPassEvent,
                SpinupState.HistoricalEvent),
            SpinupState.AnnualProcesses),
        np.where(
            spinup_state == SpinupState.HistoricalEvent,
            SpinupState.AnnualProcesses,
            SpinupState.End))


def spinup(pools, model_state, model_params):

    while True:
        model_state.spinup_state = advance_spinup_state(
            spinup_state=model_state.spinup_state,
            age=model_state.age,
            return_interval=model_params.return_interval,
            rotation_num=model_state.rotation_num,
            max_rotations=model_params.max_rotations,
            last_rotation_slow=model_state.last_rotation_slow,
            this_rotation_slow=model_state.this_rotation_slow)

        annual_process_dynamics()


def step(model_context):
    dynamics = annual_process_dynamics(
        model_context.state, model_context.params)
    matrices = expand_matrix(get_annual_process_matrix(dynamics))

    model_context.pools = compute_pools(
        model_context.dll,
        model_context.pools,
        [matrices],
        np.array(
            [list(range(0, len(model_context.params.index)))],
            dtype=np.uintp).T)
    model_context.state.age += 1


def compute_pools(dll, pools, ops, op_indices):
    pools = pools.copy()

    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        # The set op function accepts a matrix of coordinate triples.
        # In LibCBM matrices are stored in a sparse format, so 0 values can be
        # omitted from the parameter
        dll.set_op(op_id, [x for x in op],
                   np.ascontiguousarray(op_indices[:, i]))

    dll.compute_pools(op_ids, pools)

    return pools


def build_merch_vol_lookup(merch_volume):
    merch_vol_lookup = {int(i): {} for i in merch_volume.index}
    for _, row in merch_volume.iterrows():
        merch_vol_lookup[int(row.name)][int(row.age)] = float(row.volume)
    return merch_vol_lookup


def get_merch_vol(merch_vol_lookup, age, merch_vol_id):
    output = np.zeros(shape=age.shape)
    for i, age in np.ndenumerate(age):
        output[i] = merch_vol_lookup[merch_vol_id[i]][age]
    return output


def initialize(decay_parameter, disturbance_matrix, moss_c_parameter,
               inventory, mean_annual_temperature, merch_volume,
               spinup_parameter):

    libcbm_config = {
            "pools": [
                {'name': x, 'id': i+1, 'index': i}
                for i, x in enumerate(Pool.__members__.keys())],
            "flux_indicators": []
        }
    merch_vol_lookup = build_merch_vol_lookup(merch_volume)
    pools = np.zeros(shape=(len(inventory.index), len(Pool)))
    pools[:, Pool.Input] = 1.0

    max_vols = pd.DataFrame(
        {"max_merch_vol": merch_volume.volume.groupby(
            by=merch_volume.index).max()})

    dynamics_param = inventory \
        .merge(moss_c_parameter, left_on="moss_c_parameter_id",
               right_index=True, validate="m:1") \
        .merge(decay_parameter, left_on="decay_parameter_id", right_index=True,
               validate="m:1") \
        .merge(mean_annual_temperature, left_on="mean_annual_temperature_id",
               right_index=True, validate="m:1") \
        .merge(spinup_parameter, left_on="spinup_parameter_id",
               right_index=True, validate="m:1") \
        .merge(max_vols, left_on=["merch_volume_id"],
               right_index=True, validate="m:1")

    if (dynamics_param.index != inventory.index).any():
        raise ValueError()

    model_state = SimpleNamespace(
        age=inventory.age.to_numpy(),
        merch_vol=get_merch_vol(
            merch_vol_lookup,
            inventory.age.to_numpy(),
            inventory.merch_volume_id.to_numpy()))

    return SimpleNamespace(
        dll=LibCBMWrapper(
            LibCBMHandle(
                resources.get_libcbm_bin_path(),
                json.dumps(libcbm_config))),
        params=dynamics_param,
        state=model_state,
        pools=pools
    )
