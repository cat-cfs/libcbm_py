from enum import IntEnum
import numpy as np
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


class OpProcesses(IntEnum):
    """
    grouping categories for flux indicators
    """

    growth = 1
    decay = 2
    disturbance = 3


def disturbance(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def net_growth(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def overmature_decline(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def snag_turnover(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def biomass_turnover(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def dom_decay(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def slow_decay(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def slow_mixing(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    rate: float = parameters.get_slow_mixing_rate()
    op = model.create_operation(
        matrices=[
            ["AboveGroundSlowSoil", "BelowGroundSlowSoil", rate],
            ["AboveGroundSlowSoil", "AboveGroundSlowSoil", 1 - rate],
        ],
        fmt="repeating_coordinates",
    )
    # create a single matrix shared by all stands to compute slow mixing
    op.set_matrix_index(np.zeros(cbm_vars.pools.nrows))
