from __future__ import annotations
from enum import IntEnum
import numpy as np
import numba
from numba.core import types
from numba.typed import Dict
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.storage.series import Series
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn import cbm_exn_growth_functions


class OpProcesses(IntEnum):
    """
    grouping categories for flux indicators
    """

    growth = 1
    decay = 2
    disturbance = 3


class MatrixOps:
    """
    Computes and caches C flow matrices for libcbm cbm_exn C operations
    """

    def __init__(self, model: CBMModel, parameters: CBMEXNParameters):
        """initialize a MatrixOps object

        Args:
            model (CBMModel): CBMModel object
            parameters (CBMEXNParameters): constant cbm parameters
        """
        self._parameters = parameters

        self._model = model
        self._turnover_parameters = self._parameters.get_turnover_parameters()
        self._turnover_parameter_rates: dict[str, np.ndarray] = None
        self._slow_mixing_rate: float = self._parameters.get_slow_mixing_rate()

        self._turnover_parameter_idx: Dict = Dict.empty(
            key_type=types.int64,
            value_type=numba.types.DictType(types.int64, types.int64),
        )
        self._get_turnover_rates()
        self._biomass_turnover_op: Operation = None
        self._snag_turnover_op: Operation = None
        self._slow_mixing_op: Operation = None
        self._net_growth_op: Operation = None
        self._overmature_decline_op: Operation = None
        self._spinup_net_growth_op: Operation = None
        self._spinup_overmature_decline_op: Operation = None
        self._disturbance_op: Operation = None
        self._dm_index: Dict = None

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
        duplicates = self._turnover_parameters[
            ["spatial_unit_id", "sw_hw"]
        ].duplicated()
        if duplicates.any():
            duplicate_values = self._turnover_parameters[
                ["spatial_unit_id", "sw_hw"]
            ][duplicates]
            raise ValueError(
                "duplicated spuids detected in turnover paramters "
                f"{str(duplicate_values)}"
            )

        self._turnover_parameters = self._turnover_parameters.reset_index(
            drop=True
        )
        for idx, row in self._turnover_parameters.iterrows():
            spuid = int(row["spatial_unit_id"])
            sw_hw = int(row["sw_hw"])
            if spuid in self._turnover_parameter_idx:
                self._turnover_parameter_idx[spuid][sw_hw] = int(idx)
            else:
                self._turnover_parameter_idx[spuid] = Dict.empty(
                    key_type=types.int64, value_type=types.int64
                )
                self._turnover_parameter_idx[spuid][sw_hw] = int(idx)

    def disturbance(
        self, disturbance_type: Series, spuid: Series, sw_hw: Series
    ) -> Operation:
        """
        Get disturbance operations based on the specified disturbance
        type, spatial unit id and forest type (softwood or hardwood).

        If the disturbance type is <=0 this indicates no disturbance event.

        Disturbance matrices corresponding to the specified parameters
        are retrived from default parameters configuration.

        Args:
            disturbance_type (Series): series of integer disturbance type ids
                to apply to the current simulation state in this step.
            spuid (Series): the series of spatial unit ids for spatial unit
                specific disturbance information.
            sw_hw (Series): boolean series indicating if each record is a
                hardwood or softwood forest type

        Returns:
            Operation: An initialized operation with pool C flows for each
                record
        """
        if self._disturbance_op is None:
            matrix_list, self._dm_index = _disturbance(
                self._model, self._parameters
            )

            matrix_idx = self._extract_dm_index(
                self._dm_index,
                disturbance_type.to_numpy(),
                spuid.to_numpy(),
                sw_hw.to_numpy(),
            )

            self._disturbance_op = self._model.create_operation(
                matrix_list,
                fmt="matrix_list",
                process_id=OpProcesses.disturbance,
                matrix_index=matrix_idx,
                init_value=0,
            )

        else:
            self._disturbance_op.update_index(
                self._extract_dm_index(
                    self._dm_index,
                    disturbance_type.to_numpy(),
                    spuid.to_numpy(),
                    sw_hw.to_numpy(),
                )
            )
        return self._disturbance_op

    @staticmethod
    @numba.njit()
    def _extract_dm_index(
        dm_index: Dict,
        disturbance_type: np.ndarray,
        spuid: np.ndarray,
        sw_hw: np.ndarray,
    ) -> np.ndarray:
        n_rows = disturbance_type.shape[0]
        matrix_idx = np.zeros(n_rows, dtype="uintp")
        for i in range(n_rows):
            if disturbance_type[i] > 0:
                matrix_idx[i] = dm_index[
                    (disturbance_type[i], spuid[i], sw_hw[i])
                ]
        return matrix_idx

    def dom_decay(self, mean_annual_temperature: Series) -> Operation:
        """
        Compute the decay rate flows for dead organic matter (DOM) producing
        a matrix of both DOM to DOM flows and DOM to emissions (CO2) flows

        Args:
            mean_annual_temperature (Series): series of mean annual temperature

        Returns:
            Operation: decay pool C operations for each record
        """
        dom_decay_mats = _dom_decay(
            mean_annual_temperature.to_numpy(), self._parameters
        )
        dom_decay_op = self._model.create_operation(
            dom_decay_mats,
            "repeating_coordinates",
            OpProcesses.decay,
            np.arange(0, mean_annual_temperature.length),
        )
        return dom_decay_op

    def slow_decay(self, mean_annual_temperature: Series) -> Operation:
        """
        Compute the decay rate flows for dead organic matter (DOM)
        specifically for the Slow AG and Slow BG transfers to atmosphere.

        Args:
            mean_annual_temperature (Series): series of mean annual temperature

        Returns:
            Operation: decay pool C operations for each record
        """
        slow_decay_mats = _slow_decay(
            mean_annual_temperature.to_numpy(), self._parameters
        )
        slow_decay_op = self._model.create_operation(
            slow_decay_mats,
            "repeating_coordinates",
            OpProcesses.decay,
            np.arange(0, mean_annual_temperature.length),
        )
        return slow_decay_op

    def slow_mixing(self, n_rows: int) -> Operation:
        """
        Create slow mixing operations.

        Args:
            n_rows (int): number of records

        Returns:
            Operation: slow mixing operation for all records
        """
        if not self._slow_mixing_op:
            slow_mixing_mat = _slow_mixing(self._slow_mixing_rate)
            self._slow_mixing_op = self._model.create_operation(
                slow_mixing_mat,
                "repeating_coordinates",
                OpProcesses.decay,
                np.zeros(n_rows, dtype="int"),
            )
        else:
            self._slow_mixing_op.update_index(np.zeros(n_rows, dtype="int"))
        return self._slow_mixing_op

    @staticmethod
    @numba.njit()
    def _nb_turnover_matrix_idx(
        turnover_parameter_idx: Dict, spuid: np.ndarray, sw_hw: np.ndarray
    ) -> np.ndarray:
        n_rows = spuid.shape[0]
        matrix_idx = np.zeros(n_rows, dtype="uintp")
        for i in range(n_rows):
            matrix_idx[i] = turnover_parameter_idx[int(spuid[i])][
                int(sw_hw[i])
            ]
        return matrix_idx

    def _turnover_matrix_index(
        self, spuid: Series, sw_hw: Series
    ) -> np.ndarray:
        return self._nb_turnover_matrix_idx(
            self._turnover_parameter_idx, spuid.to_numpy(), sw_hw.to_numpy()
        )

    def snag_turnover(self, spuid: Series, sw_hw: Series) -> Operation:
        """Get snag turnover operations

        Args:
            spuid (Series): the series of spatial unit ids for spatial unit
                specific turnover information.
            sw_hw (Series): boolean series indicating sw or hw for forest type
                specific turnover parameters

        Returns:
            Operation: snag turnover operations for all records
        """
        if not self._snag_turnover_op:
            snag_turnover_mats = _snag_turnover(self._turnover_parameter_rates)
            self._snag_turnover_op = self._model.create_operation(
                snag_turnover_mats,
                fmt="repeating_coordinates",
                matrix_index=self._turnover_matrix_index(spuid, sw_hw),
                process_id=OpProcesses.growth,
            )

        else:
            self._snag_turnover_op.update_index(
                self._turnover_matrix_index(spuid, sw_hw)
            )
        return self._snag_turnover_op

    def biomass_turnover(self, spuid: Series, sw_hw: Series) -> Operation:
        """
        get biomass turnover operations

        Args:
            spuid (Series): the series of spatial unit ids for spatial unit
                specific turnover information.
            sw_hw (Series): boolean series indicating sw or hw for forest type
                specific turnover parameters

        Returns:
            Operation: biomass turnover operations for all records
        """
        if not self._biomass_turnover_op:
            biomass_turnover_ops = _biomass_turnover(
                self._turnover_parameter_rates
            )
            self._biomass_turnover_op = self._model.create_operation(
                biomass_turnover_ops,
                fmt="repeating_coordinates",
                matrix_index=self._turnover_matrix_index(spuid, sw_hw),
                process_id=OpProcesses.growth,
            )

        else:
            self._biomass_turnover_op.update_index(
                self._turnover_matrix_index(spuid, sw_hw)
            )
        return self._biomass_turnover_op

    def net_growth(
        self, cbm_vars: ModelVariables
    ) -> tuple[Operation, Operation]:
        """
        Get net growth operations, which consist of net growth increment
        and overmature decline

        Args:
            cbm_vars (ModelVariables): cbm state and variables

        Returns:
            tuple[Operation, Operation]: tuple of growth (item 1) and
                overmature decline (item 2) to apply to the current state.
        """
        growth_info = cbm_exn_growth_functions.prepare_growth_info(
            cbm_vars, self._parameters
        )
        self._net_growth_op = self._model.create_operation(
            _net_growth(growth_info),
            fmt="repeating_coordinates",
            matrix_index=np.arange(0, cbm_vars["pools"].n_rows),
            process_id=OpProcesses.growth,
        )

        self._overmature_decline_op = self._model.create_operation(
            _overmature_decline(growth_info),
            fmt="repeating_coordinates",
            matrix_index=np.arange(0, cbm_vars["pools"].n_rows),
            process_id=OpProcesses.growth,
        )

        return self._net_growth_op, self._overmature_decline_op

    def _spinup_net_growth_idx(
        self, spinup_vars: ModelVariables
    ) -> np.ndarray:
        age = spinup_vars["state"]["age"].to_numpy()
        op_index = (
            np.arange(0, self._spinup_2d_shape[0]) * self._spinup_2d_shape[1]
        )
        op_index = np.where(
            age >= self._spinup_2d_shape[1],
            op_index + self._spinup_2d_shape[1] - 1,
            op_index + age,
        )
        return op_index

    def spinup_net_growth(
        self, spinup_vars: ModelVariables
    ) -> tuple[Operation, Operation]:
        """
        Get net growth operations for spinup.  Spinup increments and
        overmature declines are pre-computed for a table of C increments
        by age, stand index

        Args:
            spinup_vars (ModelVariables): cbm spinup state and variables

        Returns:
            tuple[Operation, Operation]: tuple of growth (item 1) and
                overmature decline (item 2) to apply to the current spinup step
        """
        if not self._spinup_net_growth_op:
            spinup_growth_info = (
                cbm_exn_growth_functions.prepare_spinup_growth_info(
                    spinup_vars, self._parameters
                )
            )

            for k in spinup_growth_info.keys():
                self._spinup_2d_shape = spinup_growth_info[k].shape
                spinup_growth_info[k] = spinup_growth_info[k].flatten(
                    order="C"
                )
            op_index = self._spinup_net_growth_idx(spinup_vars)
            _net_growth_matrices = _net_growth(spinup_growth_info)

            self._spinup_net_growth_op = self._model.create_operation(
                matrices=_net_growth_matrices,
                fmt="repeating_coordinates",
                matrix_index=op_index,
                process_id=OpProcesses.growth,
            )
            _overmature_decline_mats = _overmature_decline(spinup_growth_info)
            self._spinup_overmature_decline_op = self._model.create_operation(
                _overmature_decline_mats,
                fmt="repeating_coordinates",
                process_id=OpProcesses.growth,
                matrix_index=op_index,
            )

        else:
            op_index = self._spinup_net_growth_idx(spinup_vars)
            self._spinup_net_growth_op.update_index(op_index)
            self._spinup_overmature_decline_op.update_index(op_index)

        return [self._spinup_net_growth_op, self._spinup_overmature_decline_op]
