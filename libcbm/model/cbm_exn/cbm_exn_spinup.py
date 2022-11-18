from typing import Callable
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition.cbm_variables import SpinupInput
from libcbm.model.model_definition.cbm_variables import SpinupVariables


def init_cbm_vars(
    model: CBMModel, spinup_vars: SpinupVariables
) -> CBMVariables:
    pass


def spinup(
    model: CBMModel,
    input: SpinupInput,
    reporting_func: Callable[[int, SpinupVariables], None] = None,
    include_flux: bool = False,
) -> CBMVariables:
    pass
