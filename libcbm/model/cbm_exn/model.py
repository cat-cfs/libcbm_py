from typing import Callable
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_variables import SpinupInput
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step


class CBMEXNModel:
    def __init__(
        self,
        pool_config: list[str],
        flux_config: list[dict],
        spinup_func: Callable[
            ["CBMEXNModel", SpinupInput], CBMVariables
        ] = None,
        step_func: Callable[
            ["CBMEXNModel", CBMVariables], CBMVariables
        ] = None,
    ):
        self._pool_config = pool_config
        self._flux_config = flux_config
        self._spinup_func = (
            cbm_exn_spinup.spinup if not spinup_func else spinup_func
        )
        self._step_func = cbm_exn_step.step if not step_func else step_func

    def spinup(self, spinup_input: SpinupInput) -> CBMVariables:
        return self._spinup_func(spinup_input)

    def step(self, cbm_vars: CBMVariables) -> CBMVariables:
        return self._step_func(self, cbm_vars)


def initialize(
    pool_config: list[str],
    flux_config: list[dict],
    spinup_func: Callable[[CBMEXNModel, SpinupInput], CBMVariables] = None,
    step_func: Callable[[CBMEXNModel, CBMVariables], CBMVariables] = None,
) -> CBMEXNModel:
    pass
