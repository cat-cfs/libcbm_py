# moss c model implementation
#
# based on referenced work:
#
# Bona, Kelly & Shaw, Cindy & Fyles, James & Kurz, Werner. (2016).
# Modelling moss-derived carbon in upland black spruce forests.
# Canadian Journal of Forest Research. 46. 10.1139/cjfr-2015-0512.
#
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources

from libcbm.model.moss_c.pools import Pool
from libcbm.model.moss_c.pools import FLUX_INDICATORS
from libcbm.model.moss_c import model_functions
from libcbm.model.moss_c.model_functions import SpinupState


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


def f5(openness, i, j, _l):
    """Sphagnum NPP (assuming 100% ground cover), NPPsp

    i*(O(t)^2) + j*(O(t) + l

    Args:
        openness (float, np.ndarray): canopy openness
        i (float, np.ndarray): MossC i parameter,
        j (float, np.ndarray): MossC j parameter,
        _l (float, np.ndarray): MossC l parameter,

    Returns:
        np.ndarray: Sphagnum moss NPP
    """
    return i * openness ** 2.0 + j * openness + _l


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


def to_numpy_namespace(df):
    return SimpleNamespace(**{
        col: df[col].to_numpy() for col in df.columns
    })


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


def spinup(model_context):
    array_shape = model_context.state.age.shape
    spinup_state = np.full(
        array_shape, SpinupState.AnnualProcesses)
    rotation_num = np.full(array_shape, 0, dtype=int)
    last_rotation_slow = np.full(array_shape, 0.0, dtype=float)
    this_rotation_slow = np.full(array_shape, 0.0, dtype=float)

    while True:
        spinup_state = model_functions.advance_spinup_state(
            spinup_state=spinup_state,
            age=model_context.state.age,
            final_age=model_context.params.age,
            return_interval=model_context.params.return_interval,
            rotation_num=rotation_num,
            max_rotations=model_context.params.max_rotations,
            last_rotation_slow=last_rotation_slow,
            this_rotation_slow=this_rotation_slow)
        last_rotation_slow[spinup_state == SpinupState.HistoricalEvent] = \
            model_context.pools[Pool.FeatherMossSlow] + \
            model_context.pools[Pool.SphagnumMossSlow]

        break


def step(model_context):
    n_stands = len(model_context.state.age)
    dynamics = annual_process_dynamics(
        model_context.state, model_context.params)
    annual_process_matrices = model_functions.expand_matrix(
        get_annual_process_matrix(dynamics))
    annual_process_matrix_index = np.array(
        list(range(0, n_stands)), dtype=np.uintp)
    disturbance_matrices = model_context.dm_data.dm_list
    disturbance_matrix_index = model_context.disturbance_types
    flux = pd.DataFrame(
        columns=[x["name"] for x in FLUX_INDICATORS],
        data=np.zeros(shape=(n_stands, len(FLUX_INDICATORS))))
    model_context.pools = model_functions.compute(
        model_context.dll,
        model_context.pools,
        flux,
        [annual_process_matrices, disturbance_matrices],
        np.column_stack([
            annual_process_matrix_index,
            disturbance_matrix_index]))

    model_context.state.age += 1
    model_context.state.merch_vol = \
        model_functions.get_merch_vol(
            model_context.merch_vol_lookup,
            model_context.state.age,
            model_context.input_data.inventory.merch_volume_id.to_numpy())
    return flux


def initialize(decay_parameter, disturbance_matrix, moss_c_parameter,
               inventory, mean_annual_temperature, merch_volume,
               spinup_parameter):

    libcbm_config = {
        "pools": [
            {'name': p.name, 'id': int(p), 'index': p_idx}
            for p_idx, p in enumerate(Pool)],
        "flux_indicators": [
            {
                "id": f_idx + 1,
                "index": f_idx,
                "process_id": f["process_id"],
                "source_pools": [int(x) for x in f["source_pools"]],
                "sink_pools": [int(x) for x in f["sink_pools"]],
            } for f_idx, f in enumerate(FLUX_INDICATORS)]
        }
    merch_vol_lookup = model_functions.build_merch_vol_lookup(merch_volume)
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

    initial_age = np.full_like(inventory.age, 1, dtype=int)
    model_state = SimpleNamespace(
        age=initial_age,
        merch_vol=model_functions.get_merch_vol(
            merch_vol_lookup,
            initial_age,
            inventory.merch_volume_id.to_numpy()))

    return SimpleNamespace(
        dll=LibCBMWrapper(
            LibCBMHandle(
                resources.get_libcbm_bin_path(),
                json.dumps(libcbm_config))),
        params=to_numpy_namespace(dynamics_param),
        state=model_state,
        pools=pools,
        merch_vol_lookup=merch_vol_lookup,
        dm_data=model_functions.initialize_dm(disturbance_matrix),
        disturbance_types=np.full_like(inventory.age, 0, dtype=np.uintp),
        input_data=SimpleNamespace(
            decay_parameter=decay_parameter,
            disturbance_matrix=disturbance_matrix,
            moss_c_parameter=moss_c_parameter,
            inventory=inventory,
            mean_annual_temperature=mean_annual_temperature,
            merch_volume=merch_volume,
            spinup_parameter=spinup_parameter)
    )
