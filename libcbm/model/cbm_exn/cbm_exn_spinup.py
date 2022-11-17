from typing import Callable
from libcbm.model.cbm_exn.model import CBMEXNModel
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_variables import SpinupInput
from libcbm.model.cbm_exn.cbm_variables import SpinupVariables


def init_cbm_vars(
    model: CBMEXNModel, spinup_vars: SpinupVariables
) -> CBMVariables:
    pass


def spinup(
    model: CBMEXNModel,
    input: SpinupInput,
    reporting_func: Callable[[int, SpinupVariables], None] = None,
    include_flux: bool = False
) -> CBMVariables:
    pass
