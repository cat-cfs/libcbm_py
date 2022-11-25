from enum import IntEnum
import numpy as np
from libcbm.storage.dataframe import DataFrame
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn import cbm_exn_functions
from libcbm.storage.series import Series


class OpProcesses(IntEnum):
    """
    grouping categories for flux indicators
    """

    growth = 1
    decay = 2
    disturbance = 3


class MatrixOps:
    def __init__(self, model: CBMModel, parameters: CBMEXNParameters):
        self._parameters = parameters

        self._model = model
        self._turnover_parameters = self._parameters.get_turnover_parameters()
        self._turnover_parameter_rates: dict[str, np.ndarray] = None
        self._slow_mixing_rate: float = self._parameters.get_slow_mixing_rate()

        # dictionary of spuid, turnover parameter matrix index
        self._turnover_parameter_idx: dict[int, int] = None
        self._get_turnover_rates()
        self._biomass_turnover_op: Operation = None
        self._snag_turnover_op: Operation = None
        self._slow_mixing_op: Operation = None
        self._net_growth_op: Operation = None

    def _get_turnover_rates(self):
        parameter_names = [
            "StemAnnualTurnoverRate",
            "FoliageFallRate",
            "BranchTurnoverRate",
            "CoarseRootTurnProp",
            "FineRootTurnProp",
            "OtherToBranchSnagSplit",
            "CoarseRootAGSplit",
            "FineRootAGSplit",
            "StemSnag",
            "BranchSnag",
        ]
        self._turnover_parameter_rates = {
            str(col): self._turnover_parameters[col].to_numpy()
            for col in parameter_names
        }
        duplicates = self._turnover_parameters["spuid"].duplicated()
        if duplicates.any():
            duplicate_values = self._turnover_parameters["spuid"][duplicates]
            raise ValueError(
                "duplicated spuids detected in turnover paramters "
                f"{duplicate_values}"
            )
        self._turnover_parameter_idx = {
            int(spuid): int(idx)
            for idx, spuid in enumerate(
                self._turnover_parameters["spuid"].to_numpy()
            )
        }

    def snag_turnover(self, spuid: Series) -> Operation:
        if not self._snag_turnover_op:
            self._snag_turnover_op = _snag_turnover(
                self._model, self._turnover_parameter_rates
            )
            self._snag_turnover_op.set_op(
                spuid.map(self._turnover_parameter_idx)
            )
        else:
            self._snag_turnover_op.update_index(
                spuid.map(self._turnover_parameter_idx)
            )
        return self._snag_turnover_op

    def biomass_turnover(self, spuid: Series) -> Operation:
        if not self._biomass_turnover_op:
            self._biomass_turnover_op = _biomass_turnover(
                self._model, self._turnover_parameter_rates
            )
            self._biomass_turnover_op.set_op(
                spuid.map(self._turnover_parameter_idx)
            )
        else:
            self._biomass_turnover_op.update_index(
                spuid.map(self._turnover_parameter_idx)
            )
        return self._biomass_turnover_op

    def slow_mixing(self, n_rows: int) -> Operation:
        if not self._slow_mixing_op:
            self._slow_mixing_op = _slow_mixing(
                self._model, self._slow_mixing_rate
            )
            self._slow_mixing_op.set_op(np.zeros(n_rows, dtype="int"))
        else:
            self._slow_mixing_op.update_index(np.zeros(n_rows, dtype="int"))
        return self._slow_mixing_op

    def net_growth(
        self, cbm_vars: CBMVariables
    ) -> tuple[Operation, Operation]:
        if not self._net_growth_op:
            cbm_exn_functions.prepare_growth_info(cbm_vars)
            pass
        else:
            self._net_growth_op.update_index(
                np.arange(0, cbm_vars["pools"].n_rows, dtype="int")
            )
            return self._net_growth_op


def _disturbance(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def _net_growth(
    model: CBMModel,
    growth_info: DataFrame,
) -> Operation:
    op = model.create_operation(
        matrices=[
            ["Input", "Merch", growth_info["merch_inc"].to_numpy()],
            ["Input", "Other", growth_info["other_inc"].to_numpy()],
            ["Input", "Foliage", growth_info["foliage_inc"].to_numpy()],
            [
                "Input",
                "CoarseRoots",
                growth_info["coarse_root_inc"].to_numpy(),
            ],
            ["Input", "FineRoots", growth_info["fine_root_inc"].to_numpy()],
        ]
    )
    return op


def _overmature_decline(
    model: CBMModel,
    merch_loss: np.ndarray,
    other_loss: np.ndarray,
    foliage_loss: np.ndarray,
    coarse_root_loss: np.ndarray,
    fine_root_loss: np.ndarray,
    turnover_rates: dict[str, np.ndarray],
) -> Operation:

    op = model.create_operation(
        matrices=[
            ["Merch", "StemSnag", merch_loss],
            [
                "Other",
                "BranchSnag",
                other_loss * turnover_rates["OtherToBranchSnagSplit"],
            ],
            [
                "Other",
                "AboveGroundFastSoil",
                other_loss * (1 - turnover_rates["OtherToBranchSnagSplit"]),
            ],
            ["Foliage", "AboveGroundVeryFastSoil", foliage_loss],
            [
                "CoarseRoots",
                "AboveGroundFastSoil",
                coarse_root_loss * turnover_rates["CoarseRootAGSplit"],
            ],
            [
                "CoarseRoots",
                "BelowGroundFastSoil",
                coarse_root_loss * (1 - turnover_rates["CoarseRootAGSplit"]),
            ],
            [
                "FineRoots",
                "AboveGroundVeryFastSoil",
                fine_root_loss * turnover_rates["FineRootAGSplit"],
            ],
            [
                "FineRoots",
                "BelowGroundVeryFastSoil",
                fine_root_loss * (1 - turnover_rates["FineRootAGSplit"]),
            ],
        ]
    )
    return op


def _snag_turnover(model: CBMModel, rates: dict[str, np.ndarray]) -> Operation:

    op = model.create_operation(
        matrices=[
            ["StemSnag", "StemSnag", 1 - rates["StemSnag"]],
            ["StemSnag", "MediumSoil", rates["StemSnag"]],
            ["BranchSnag", "BranchSnag", 1 - rates["BranchSnag"]],
            ["BranchSnag", "AboveGroundFastSoil", rates["BranchSnag"]],
        ],
        fmt="repeating_coordinates",
        process_id=OpProcesses.growth,
    )
    return op


def _biomass_turnover(
    model: CBMModel, rates: dict[str, np.ndarray]
) -> Operation:

    op = model.create_operation(
        matrices=[
            [
                "Merch",
                "SoftwoodStemSnag",
                rates["StemAnnualTurnoverRate"],
            ],
            [
                "Foliage",
                "AboveGroundVeryFastSoil",
                rates["FoliageFallRate"],
            ],
            [
                "Other",
                "SoftwoodBranchSnag",
                rates["OtherToBranchSnagSplit"] * rates["BranchTurnoverRate"],
            ],
            [
                "Other",
                "AboveGroundFastSoil",
                (1 - rates["OtherToBranchSnagSplit"])
                * rates["BranchTurnoverRate"],
            ],
            [
                "CoarseRoots",
                "AboveGroundFastSoil",
                rates["CoarseRootAGSplit"] * rates["CoarseRootTurnProp"],
            ],
            [
                "CoarseRoots",
                "BelowGroundFastSoil",
                (1 - rates["CoarseRootAGSplit"]) * rates["CoarseRootTurnProp"],
            ],
            [
                "FineRoots",
                "AboveGroundVeryFastSoil",
                rates["FineRootAGSplit"] * rates["FineRootTurnProp"],
            ],
            [
                "FineRoots",
                "BelowGroundVeryFastSoil",
                (1 - rates["FineRootAGSplit"]) * rates["FineRootTurnProp"],
            ],
        ],
        fmt="repeating_coordinates",
        process_id=OpProcesses.growth,
    )
    return op


def _dom_decay(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def _slow_decay(
    model: CBMModel, cbm_vars: CBMVariables, parameters: CBMEXNParameters
) -> Operation:
    pass


def _slow_mixing(model: CBMModel, rate: float) -> Operation:

    op = model.create_operation(
        matrices=[
            ["AboveGroundSlowSoil", "BelowGroundSlowSoil", rate],
            ["AboveGroundSlowSoil", "AboveGroundSlowSoil", 1 - rate],
        ],
        fmt="repeating_coordinates",
        process_id=OpProcesses.decay,
    )
    return op
