import numpy as np
from libcbm.model.cbm_exn.model import CBMEXNModel
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


def net_growth(model: CBMEXNModel, cbm_vars: CBMVariables) -> Operation:
    pass


def overmature_decline(
    model: CBMEXNModel, cbm_vars: CBMVariables
) -> Operation:
    pass


def snag_turnover(model: CBMEXNModel, cbm_vars: CBMVariables) -> Operation:
    pass


def dom_decay(model: CBMEXNModel, cbm_vars: CBMVariables) -> Operation:
    pass


def slow_decay(model: CBMEXNModel, cbm_vars: CBMVariables) -> Operation:
    pass


def slow_mixing(model: CBMEXNModel, cbm_vars: CBMVariables) -> Operation:
    parameters = CBMEXNParameters(model.parameters)
    rate: float = parameters.get_slow_mixing_rate()
    op = model._model_handle.create_operation(
        matrices=[
            ["AboveGroundSlowSoil", "BelowGroundSlowSoil", rate],
            ["AboveGroundSlowSoil", "AboveGroundSlowSoil", 1 - rate],
        ],
        fmt="repeating_coordinates",
    )
    # create a single matrix shared by all stands to compute slow mixing
    op.set_matrix_index(np.zeros(cbm_vars.pools.nrows))
