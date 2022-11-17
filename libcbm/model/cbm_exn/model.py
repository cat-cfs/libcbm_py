from typing import Callable
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_variables import SpinupInput


class CBMEXNModel:
    pass


def initialize(
    pool_config: list[str],
    flux_config: list[dict],
    spinup: Callable[[CBMEXNModel, SpinupInput], CBMVariables] = None,
    step: Callable[[CBMEXNModel, CBMVariables], CBMVariables] = None,
):
    pass
