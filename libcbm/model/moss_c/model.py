# moss c model implementation
#
# based on referenced work:
#
# Bona, Kelly & Shaw, Cindy & Fyles, James & Kurz, Werner. (2016).
# Modelling moss-derived carbon in upland black spruce forests.
# Canadian Journal of Forest Research. 46. 10.1139/cjfr-2015-0512.
#

from typing import Union
from types import SimpleNamespace
import numba
import numpy as np


from libcbm.model.moss_c.pools import Pool
from libcbm.model.moss_c.pools import ANNUAL_PROCESSES
from libcbm.model.moss_c.pools import DISTURBANCE_PROCESS
from libcbm.model.model_definition.spinup_engine import SpinupState
from libcbm.model.model_definition import spinup_engine
from libcbm.model.moss_c.model_context import ModelContext
from libcbm.wrapper import libcbm_operation
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
from libcbm.storage.series import SeriesDef
from libcbm.storage import series


def f1(
    merch_vol: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
) -> np.ndarray:
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
        merch_vol == 0,
        60,
        np.power(
            10,
            a * np.log10(merch_vol, where=(merch_vol != 0)) + b,
            where=(merch_vol != 0),
        ),
    )


def f2(
    openness: Union[float, np.ndarray],
    stand_age: Union[float, np.ndarray],
    c: Union[float, np.ndarray],
    d: Union[float, np.ndarray],
) -> np.ndarray:
    """returns Feather moss ground cover, GCfm(t)

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
        stand_age < 10, 0.0, np.where(openness > 70.0, 100.0, openness * c + d)
    )


def f3(
    openness: Union[float, np.ndarray],
    stand_age: Union[float, np.ndarray],
    e: Union[float, np.ndarray],
    f: Union[float, np.ndarray],
) -> np.ndarray:
    """returns Sphagnum moss ground cover, GCsp(t)

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
        stand_age < 10, 0.0, np.where(openness > 70.0, 100.0, openness * e + f)
    )


def f4(
    openness: Union[float, np.ndarray],
    g: Union[float, np.ndarray],
    h: Union[float, np.ndarray],
) -> np.ndarray:
    """Feathermoss NPP (assuming 100% ground cover), NPPfm

    g*O(t)^h

    Args:
        openness (float, np.ndarray): canopy openness
        g (float, np.ndarray): MossC g parameter,
        h (float, np.ndarray): MossC h parameter,
    """
    return np.where(openness < 5.0, 0.6, g * np.power(openness, h))


def f5(
    openness: Union[float, np.ndarray],
    i: Union[float, np.ndarray],
    j: Union[float, np.ndarray],
    _l: Union[float, np.ndarray],
) -> np.ndarray:
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
    return i * openness**2.0 + j * openness + _l


def f6(
    merch_vol: Union[float, np.ndarray],
    m: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
) -> np.ndarray:
    """

    Args:
        merch_vol (float, np.ndarray): merchantable volume.
        m (float, np.ndarray): MossC m parameter,
        n (float, np.ndarray): MossC n parameter,
    """
    return np.log(merch_vol) * m + n


def f7(
    mean_annual_temp: Union[float, np.ndarray],
    base_decay_rate: Union[float, np.ndarray],
    q10: Union[float, np.ndarray],
    t_ref: Union[float, np.ndarray],
) -> np.ndarray:
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
        (mean_annual_temp - t_ref) * np.log(q10) * 0.1
    )


class AnnualProcessDynamics:
    def __init__(
        self,
        kss: Union[float, np.ndarray],
        openness: Union[float, np.ndarray],
        akff: Union[float, np.ndarray],
        akfs: Union[float, np.ndarray],
        aksf: Union[float, np.ndarray],
        akss: Union[float, np.ndarray],
        GCfm: Union[float, np.ndarray],
        GCsp: Union[float, np.ndarray],
        NPPfm: Union[float, np.ndarray],
        NPPsp: Union[float, np.ndarray],
    ):
        self._kss = kss
        self._openness = openness
        self._akff = akff
        self._akfs = akfs
        self._aksf = aksf
        self._akss = akss
        self._GCfm = GCfm
        self._GCsp = GCsp
        self._NPPfm = NPPfm
        self._NPPsp = NPPsp

    @property
    def kss(self) -> Union[float, np.ndarray]:
        return self._kss

    @property
    def openness(self) -> Union[float, np.ndarray]:
        return self._openness

    @property
    def akff(self) -> Union[float, np.ndarray]:
        return self._akff

    @property
    def akfs(self) -> Union[float, np.ndarray]:
        return self._akfs

    @property
    def aksf(self) -> Union[float, np.ndarray]:
        return self._aksf

    @property
    def akss(self) -> Union[float, np.ndarray]:
        return self._akss

    @property
    def GCfm(self) -> Union[float, np.ndarray]:
        return self._GCfm

    @property
    def GCsp(self) -> Union[float, np.ndarray]:
        return self._GCsp

    @property
    def NPPfm(self) -> Union[float, np.ndarray]:
        return self._NPPfm

    @property
    def NPPsp(self) -> Union[float, np.ndarray]:
        return self._NPPsp


def annual_process_dynamics(
    state: DataFrame, params: DataFrame
) -> AnnualProcessDynamics:

    _p = SimpleNamespace(
        **{col: params[col].to_numpy() for col in params.columns}
    )
    _s = SimpleNamespace(
        **{col: state[col].to_numpy() for col in state.columns}
    )

    kss = f6(_p.max_merch_vol, _p.m, _p.n)
    openness = f1(_s.merch_vol, _p.a, _p.b)

    return AnnualProcessDynamics(
        kss=kss,
        openness=openness,
        # applied feather moss fast pool decay rate
        akff=f7(_p.mean_annual_temp, _p.kff, _p.q10, _p.tref),
        # applied feather moss slow pool decay rate
        akfs=f7(_p.mean_annual_temp, _p.kfs, _p.q10, _p.tref),
        # applied sphagnum fast pool applied decay rate
        aksf=f7(_p.mean_annual_temp, _p.ksf, _p.q10, _p.tref),
        # applied sphagnum slow pool applied decay rate
        akss=f7(_p.mean_annual_temp, kss, _p.q10, _p.tref),
        # Feather moss ground cover
        GCfm=f2(openness, _s.age, _p.c, _p.d),
        # Sphagnum ground cover
        GCsp=f3(openness, _s.age, _p.e, _p.f),
        # Feathermoss NPP (assuming 100% ground cover)
        NPPfm=f4(openness, _p.g, _p.h),
        # Sphagnum NPP (assuming 100% ground cover)
        NPPsp=f5(openness, _p.i, _p.j, _p.l),
    )


def get_annual_process_matrix(dynamics_param: AnnualProcessDynamics) -> list:
    mat = [
        [
            Pool.Input,
            Pool.FeatherMossLive,
            dynamics_param.NPPfm * dynamics_param.GCfm / 100.0,
        ],
        [
            Pool.Input,
            Pool.SphagnumMossLive,
            dynamics_param.NPPsp * dynamics_param.GCsp / 100.0,
        ],
        # turnovers
        [Pool.FeatherMossLive, Pool.FeatherMossFast, 1.0],
        [Pool.FeatherMossLive, Pool.FeatherMossLive, 0.0],
        [
            Pool.FeatherMossFast,
            Pool.FeatherMossSlow,
            dynamics_param.akff * 0.15,
        ],
        [Pool.SphagnumMossLive, Pool.SphagnumMossFast, 1.0],
        [Pool.SphagnumMossLive, Pool.SphagnumMossLive, 0.0],
        [
            Pool.SphagnumMossFast,
            Pool.SphagnumMossSlow,
            dynamics_param.aksf * 0.15,
        ],
        # fast losses
        [
            Pool.FeatherMossFast,
            Pool.FeatherMossFast,
            1.0 - dynamics_param.akff,
        ],
        [
            Pool.SphagnumMossFast,
            Pool.SphagnumMossFast,
            1.0 - dynamics_param.aksf,
        ],
        # decays
        [Pool.FeatherMossFast, Pool.CO2, dynamics_param.akff * 0.85],
        [Pool.SphagnumMossFast, Pool.CO2, dynamics_param.aksf * 0.85],
        [Pool.FeatherMossSlow, Pool.CO2, dynamics_param.akfs],
        [
            Pool.FeatherMossSlow,
            Pool.FeatherMossSlow,
            1.0 - dynamics_param.akfs,
        ],
        [Pool.SphagnumMossSlow, Pool.CO2, dynamics_param.akss],
        [
            Pool.SphagnumMossSlow,
            Pool.SphagnumMossSlow,
            1.0 - dynamics_param.akss,
        ],
    ]
    return mat


@numba.njit()
def update_spinup_variables(
    n_stands: int,
    spinup_state: np.ndarray,
    dist_type: np.ndarray,
    pools: np.ndarray,
    last_rotation_slow: np.ndarray,
    this_rotation_slow: np.ndarray,
    rotation_num: np.ndarray,
    historical_dist_type: np.ndarray,
    last_pass_dist_type: np.ndarray,
    enabled: np.ndarray,
) -> bool:
    enabled_count = n_stands
    for i in range(n_stands):
        state = spinup_state[i]
        if state == SpinupState.LastPassEvent:
            dist_type[i] = last_pass_dist_type[i]
        elif state == SpinupState.HistoricalEvent:
            dist_type[i] = historical_dist_type[i]
            last_rotation_slow[i] = (
                pools[i, Pool.SphagnumMossSlow.value]
                + pools[i, Pool.FeatherMossSlow.value]
            )
            rotation_num[i] += 1
        else:
            if state == SpinupState.End:
                enabled[i] = 0
                enabled_count -= 1
            dist_type[i] = 0
            this_rotation_slow[i] = (
                pools[i, Pool.SphagnumMossSlow.value]
                + pools[i, Pool.FeatherMossSlow.value]
            )
    return enabled_count == 0


class SpinupDebug:
    def __init__(self):
        self.pools = None
        self.state = None
        self.model_context = None
        self.spinup_vars = None

    def append_spinup_debug_record(
        self,
        iteration: int,
        model_context: ModelContext,
        spinup_vars: DataFrame,
    ):
        state_t = model_context.state.copy()
        state_t.add_column(
            series.allocate(
                "t",
                state_t.n_rows,
                iteration,
                "int",
                state_t.backend_type,
            ),
            index=0,
        )
        self.state = dataframe.concat_data_frame([self.state, state_t])

        pools_t = model_context.pools.copy()
        pools_t.add_column(
            series.allocate(
                "t",
                pools_t.n_rows,
                iteration,
                "int",
                pools_t.backend_type,
            ),
            index=0,
        )
        self.pools = dataframe.concat_data_frame([self.pools, pools_t])

        spinup_vars_t = spinup_vars.copy()
        spinup_vars_t.add_column(
            series.allocate(
                "t",
                spinup_vars_t.n_rows,
                iteration,
                "int",
                spinup_vars_t.backend_type,
            ),
            index=0,
        )
        self.spinup_vars = dataframe.concat_data_frame(
            [self.spinup_vars, spinup_vars_t]
        )


def spinup(
    model_context: ModelContext, enable_debugging: bool = False
) -> Union[None, SpinupDebug]:

    if enable_debugging:
        spinup_debug = SpinupDebug()
    else:
        spinup_debug = None

    spinup_vars: DataFrame = dataframe.from_series_list(
        [
            SeriesDef("spinup_state", SpinupState.AnnualProcesses, "int"),
            SeriesDef("rotation_num", 0, "int"),
            SeriesDef("last_rotation_slow", 0.0, "float"),
            SeriesDef("this_rotation_slow", 0.0, "float"),
        ],
        nrows=model_context.inventory.n_rows,
        back_end=model_context.backend_type,
    )
    iteration = 0
    while True:

        state = spinup_engine.advance_spinup_state(
            spinup_state=spinup_vars["spinup_state"],
            age=model_context.state["age"],
            final_age=model_context.parameters["age"],
            return_interval=model_context.parameters["return_interval"],
            rotation_num=spinup_vars["rotation_num"],
            max_rotations=model_context.parameters["max_rotations"],
            last_rotation_slow=spinup_vars["last_rotation_slow"],
            this_rotation_slow=spinup_vars["this_rotation_slow"],
        )
        spinup_vars["spinup_state"].assign(state)

        all_finished = update_spinup_variables(
            n_stands=model_context.n_stands,
            spinup_state=spinup_vars["spinup_state"].to_numpy(),
            dist_type=model_context.state["disturbance_type"].to_numpy(),
            pools=model_context.pools.to_numpy(),
            last_rotation_slow=spinup_vars["last_rotation_slow"].to_numpy(),
            this_rotation_slow=spinup_vars["this_rotation_slow"].to_numpy(),
            rotation_num=spinup_vars["rotation_num"].to_numpy(),
            historical_dist_type=model_context.inventory[
                "historical_dm_index"
            ].to_numpy(),
            last_pass_dist_type=model_context.inventory[
                "last_pass_dm_index"
            ].to_numpy(),
            enabled=model_context.state["enabled"].to_numpy(),
        )
        if all_finished:
            # re-enable everything for subsequent processes
            model_context.state["enabled"].assign(1)
            break
        step(
            model_context,
            disturbance_before_annual_process=False,
            include_flux=False,
        )
        if enable_debugging:
            spinup_debug.append_spinup_debug_record(
                iteration, model_context, spinup_vars
            )
        iteration += 1

    return spinup_debug


def step(
    model_context: ModelContext,
    disturbance_before_annual_process: bool = True,
    include_flux: bool = True,
) -> None:
    n_stands = model_context.state["age"].length
    model_context.state["merch_vol"].assign(
        model_context.merch_vol_lookup.get_merch_vol(
            model_context.state["age"],
            model_context.parameters["merch_volume_id"],
        )
    )
    dynamics_param = annual_process_dynamics(
        model_context.state, model_context.parameters
    )
    annual_process_matrix = get_annual_process_matrix(dynamics_param)
    annual_process_matrices = libcbm_operation.Operation(
        model_context.dll,
        libcbm_operation.OperationFormat.RepeatingCoordinates,
        annual_process_matrix,
    )
    annual_process_matrices.set_op(
        np.array(list(range(0, n_stands)), dtype=np.uintp)
    )
    disturbance_matrices = libcbm_operation.Operation(
        model_context.dll,
        libcbm_operation.OperationFormat.MatrixList,
        model_context.disturbance_matrices.dm_list,
    )
    disturbance_matrices.set_op(
        model_context.state["disturbance_type"].to_numpy()
    )

    flux = None
    if include_flux:
        model_context.initialize_flux()
        flux = model_context.flux

    if disturbance_before_annual_process:
        op_processes = [DISTURBANCE_PROCESS, ANNUAL_PROCESSES]
        ops = [disturbance_matrices, annual_process_matrices]
    else:
        op_processes = [ANNUAL_PROCESSES, DISTURBANCE_PROCESS]
        ops = [annual_process_matrices, disturbance_matrices]

    libcbm_operation.compute(
        dll=model_context.dll,
        pools=model_context.pools,
        operations=ops,
        op_processes=op_processes,
        flux=flux,
        enabled=model_context.state["enabled"],
    )

    for op in ops:
        op.dispose()

    age_zero_indices = dataframe.indices_nonzero(
        (model_context.state["disturbance_type"] != 0)
        & (model_context.state["enabled"] != 0)
    )
    age_increment_indices = dataframe.indices_nonzero(
        (model_context.state["disturbance_type"] == 0)
        & (model_context.state["enabled"] != 0)
    )
    model_context.state["age"].assign(
        model_context.state["age"].take(age_increment_indices) + 1,
        age_increment_indices,
    )
    model_context.state["age"].assign(0, age_zero_indices)
