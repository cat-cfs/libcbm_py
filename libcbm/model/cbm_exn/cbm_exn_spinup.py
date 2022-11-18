from typing import Callable
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables


def init_cbm_vars(model: CBMModel, spinup_vars: CBMVariables) -> CBMVariables:
    pass


def spinup(
    model: CBMModel,
    input: CBMVariables,
    reporting_func: Callable[[int, CBMVariables], None] = None,
    include_flux: bool = False,
) -> CBMVariables:
    pass
