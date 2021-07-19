from enum import IntEnum
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse

from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources

class Pool(IntEnum):
    Input=0,
    FeatherMossLive=1,
    SphagnumMossLive=2,
    FeatherMossFast=3,
    SphagnumMossFast=4,
    FeatherMossSlow=5,
    SphagnumMossSlow=6,
    CO2=7
   

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
    return np.where(
        merch_vol == 0.0, 60.0,
        np.power(10, a * np.log10(merch_vol) + b))

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
        merch_vol ([type]): [description]
        m ([type]): [description]
        n ([type]): [description]
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


def c_annual_process_dynamics_params(df_params):

    kss = f6(df_params.max_merch_vol, df_params.m, df_params.n)
    openness = f1(df_params.merch_vol, df_params.a, df_params.b)
    
    return SimpleNamespace(
        kss=kss,
        opennes=openness,
    
        # applied feather moss fast pool decay rate
        akff=f7(
            df_params.mean_annual_temp, df_params.kff, df_params.q10,
            df_params.tref),

        # applied feather moss slow pool decay rate
        akfs=f7(
            df_params.mean_annual_temp, df_params.kfs, df_params.q10,
            df_params.tref), 
        
        # applied sphagnum fast pool applied decay rate
        aksf=f7(
            df_params.mean_annual_temp, df_params.ksf, df_params.q10,
            df_params.tref),

        # applied sphagnum slow pool applied decay rate
        akss=f7(
            df_params.mean_annual_temp, kss, df_params.q10, df_params.tref),

        # Feather moss ground cover
        GCfm=f2(openness, df_params.age, df_params.c, df_params.d),
        
        # Sphagnum ground cover
        GCsp=f3(openness, df_params.age, df_params.e, df_params.f),

        # Feathermoss NPP (assuming 100% ground cover)
        NPPfm=f4(openness, df_params.g, df_params.h),

        # Sphagnum NPP (assuming 100% ground cover)
        NPPsp = f5(
            openness, df_params.i, df_params.j, df_params.l))
    
def get_annual_process_matrix(dynamics_param):
    n_params = len(dynamics_param.NPPfm)

    mat = sparse.coo_matrix(shape=(len(Pool), len(Pool), n_params))
    flows = np.array([
        [Pool.Input, Pool.FeatherMossLive, NPPfm * GCfm/100.0]
        [Pool.Input, Pool.SphagnumMossLive, NPPsp * GCsp/100.0],
 
        # turnovers
        [Pool.FeatherMossLive, Pool.FeatherMossFast, 1.0],
        [Pool.FeatherMossLive, Pool.FeatherMossLive, 0.0],
 
        [Pool.FeatherMossFast, Pool.FeatherMossSlow, akff * 0.15],
 
        [Pool.SphagnumMossLive, Pool.SphagnumMossFast, 1.0],
        [Pool.SphagnumMossLive, Pool.SphagnumMossLive, 0.0],
 
        [Pool.SphagnumMossFast, Pool.SphagnumMossSlow, aksf*0.15],
 
        # fast losses
        [Pool.FeatherMossFast, Pool.FeatherMossFast, 1.0-akff],
        [Pool.SphagnumMossFast, Pool.SphagnumMossFast, 1.0-aksf],
 
        # decays
        [Pool.FeatherMossFast, Pool.CO2, akff*0.85],
 
        [Pool.SphagnumMossFast, Pool.CO2, aksf*0.85],
 
        [Pool.FeatherMossSlow, Pool.CO2, akfs],
        [Pool.FeatherMossSlow, Pool.FeatherMossSlow, 1.0-akfs],
 
        [Pool.SphagnumMossSlow, Pool.CO2, akss],
        [Pool.SphagnumMossSlow, Pool.SphagnumMossSlow, 1.0-akss]
    ])
    return flows


def get_disturbance_flows(disturbance_type_name, disturbance_matrices):
    matrix = [
        [

        ]
    ]
    return matrix


class SpinupState(IntEnum):
    AnnualProcesses=1,
    HistoricalEvent=2,
    LastPassEvent=3,
    End=4


def _small_slow_diff(last_rotation_slow, this_rotation_slow):
    return abs((last_rotation_slow - this_rotation_slow) \
           / (last_rotation_slow+this_rotation_slow)/2.0) < 0.001


def advance_spinup_state(spinup_state, age, return_interval, rotation_num, max_rotations, last_rotation_slow, this_rotation_slow):
    np.where(
        spinup_state == SpinupState.AnnualProcesses, 
        np.where(
            age >= return_interval,
            np.where(
                _small_slow_diff(last_rotation_slow, this_rotation_slow)
                | rotation_num > max_rotations,
                SpinupState.LastPassEvent,
                SpinupState.HistoricalEvent)),
            SpinupState.AnnualProcesses,
        np.where(
            spinup_state == SpinupState.HistoricalEvent,
            SpinupState.AnnualProcesses,
            SpinupState.End))


    #if spinup_state == SpinupState.AnnualProcesses:
    #    if age >= return_interval:
    #        small_slow_diff = \
    #            abs((last_rotation_slow - this_rotation_slow) \
    #            / (last_rotation_slow+this_rotation_slow)/2.0) < 0.001
    #        if small_slow_diff or rotation_num >= max_rotations:
    #            return SpinupState.LastPassEvent
    #        else:
    #            return SpinupState.HistoricalEvent
    #    else:
    #        return SpinupState.AnnualProcesses
    #elif spinup_state == SpinupState.HistoricalEvent:
    #    return SpinupState.AnnualProcesses
    #elif spinup_state == SpinupState.LastPassEvent:
    #    return SpinupState.End

def get_ops(model_state):
    if model_state.spinup_state == SpinupState.AnnualProcesses:
        mat_index = get_annual_process_matrix()

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
        
        
        annual_process_flows = []
        for row_idx in range(0, len(model_state.index)):
            annual_process_flows.append(
                c_annual_process_dynamics(
                   # row.iloc[].age,
                   # paramsmean_annual_temp, max_merch_vol, merch_vol,
                   #           function_params, decay_params
                ))


def compute_pools(dll, pools, ops, op_indices):
    pools = pools.copy()
    
    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        #The set op function accepts a matrix of coordinate triples.  
        #In LibCBM matrices are stored in a sparse format, so 0 values can be omitted from the parameter
        dll.set_op(op_id, [to_coordinate(x) for x in op], 
                   np.ascontiguousarray(op_indices[:,i]))
        
    dll.compute_pools(op_ids, pools)
    
    return pools


def initialize(config):
    dll = LibCBMWrapper(
        LibCBMHandle(
            resources.get_libcbm_bin_path(),
            json.dumps(config)))

def run():
    pools = np.repeat()
    flux_processor = initialize(
        config={
            "pools": [
                {'name': x, 'id': i+1, 'index': i}
                 for i, x in enumerate(Pool.__members__.keys())],
            "flux_indicators": []
        })