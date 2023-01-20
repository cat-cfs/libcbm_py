from __future__ import annotations
from enum import IntEnum
import numpy as np
import numba
from numba.core import types
from numba.typed import Dict
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.model_variables import ModelVariables
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
        growth_info = cbm_exn_functions.prepare_growth_info(
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
            spinup_growth_info = cbm_exn_functions.prepare_spinup_growth_info(
                spinup_vars, self._parameters
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


def _disturbance(
    model: CBMModel, parameters: CBMEXNParameters
) -> tuple[list, Dict]:
    disturbance_matrices = parameters.get_disturbance_matrices()

    matrix_data_by_dmid: dict[int, list[list]] = {}
    dmid_index = Dict.empty(key_type=types.int64, value_type=types.int64)

    dmid = disturbance_matrices["disturbance_matrix_id"].to_numpy()
    source = disturbance_matrices["source_pool"].to_list()
    sink = disturbance_matrices["sink_pool"].to_list()
    proportion = disturbance_matrices["proportion"].to_numpy()
    for i in range(dmid.shape[0]):
        dmid_i = dmid[i]
        if dmid_i not in matrix_data_by_dmid:
            matrix_data_by_dmid[dmid_i] = []
        matrix_data_by_dmid[dmid_i].append([source[i], sink[i], proportion[i]])
    dmids = list(matrix_data_by_dmid.keys())
    for i_dmid, dmid in enumerate(dmids):
        pool_set = set(model.pool_names)
        dmid_index[dmid] = i_dmid + 1
        for row in matrix_data_by_dmid[dmid]:
            pool_set.discard(row[0])
        for pool in pool_set:
            matrix_data_by_dmid[dmid].append([pool, pool, 1.0])

    dm_assocatiations = parameters.get_disturbance_matrix_associations()
    _dm_op_index = Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.int64,
    )
    _dm_op_index = _build_dm_op_index(
        _dm_op_index,
        dmid_index,
        dm_assocatiations["disturbance_matrix_id"].to_numpy(dtype="int"),
        dm_assocatiations["disturbance_type_id"].to_numpy(dtype="int"),
        dm_assocatiations["spatial_unit_id"].to_numpy(dtype="int"),
        dm_assocatiations["sw_hw"].to_numpy(dtype="int"),
    )
    # append the null matrix
    matrix_list = [[[p, p, 1.0] for p in model.pool_names]] + list(
        matrix_data_by_dmid.values()
    )
    return (matrix_list, _dm_op_index)


@numba.njit()
def _build_dm_op_index(
    _dm_op_index: Dict,
    dmid_idx: Dict,
    disturbance_matrix_id: np.ndarray,
    disturbance_type_id: np.ndarray,
    spatial_unit_ids: np.ndarray,
    sw_hw: np.ndarray,
):

    for i in range(disturbance_type_id.shape[0]):
        dist_type = disturbance_type_id[i]
        spuid = spatial_unit_ids[i]
        _sw_hw = sw_hw[i]
        key = (dist_type, spuid, _sw_hw)
        dm_idx = dmid_idx[disturbance_matrix_id[i]]
        _dm_op_index[key] = dm_idx

    return _dm_op_index


def _net_growth(
    growth_info: dict[str, np.ndarray],
) -> list:

    matrices = [
        ["Input", "Merch", growth_info["merch_inc"] * 0.5],
        ["Input", "Other", growth_info["other_inc"] * 0.5],
        ["Input", "Foliage", growth_info["foliage_inc"] * 0.5],
        ["Input", "CoarseRoots", growth_info["coarse_root_inc"] * 0.5],
        ["Input", "FineRoots", growth_info["fine_root_inc"] * 0.5],
    ]
    return matrices


def _overmature_decline(
    growth_info: dict[str, np.ndarray],
) -> list:

    matrices = [
        ["Merch", "StemSnag", growth_info["merch_to_stem_snag_prop"]],
        ["Other", "BranchSnag", growth_info["other_to_branch_snag_prop"]],
        [
            "Other",
            "AboveGroundFastSoil",
            growth_info["other_to_ag_fast_prop"],
        ],
        [
            "Foliage",
            "AboveGroundVeryFastSoil",
            growth_info["foliage_to_ag_fast_prop"],
        ],
        [
            "CoarseRoots",
            "AboveGroundFastSoil",
            growth_info["coarse_root_to_ag_fast_prop"],
        ],
        [
            "CoarseRoots",
            "BelowGroundFastSoil",
            growth_info["coarse_root_to_bg_fast_prop"],
        ],
        [
            "FineRoots",
            "AboveGroundVeryFastSoil",
            growth_info["fine_root_to_ag_vfast_prop"],
        ],
        [
            "FineRoots",
            "BelowGroundVeryFastSoil",
            growth_info["fine_root_to_bg_vfast_prop"],
        ],
    ]
    return matrices


def _snag_turnover(rates: dict[str, np.ndarray]) -> list:
    matrices = [
        ["StemSnag", "StemSnag", 1 - rates["StemSnag"]],
        ["StemSnag", "MediumSoil", rates["StemSnag"]],
        ["BranchSnag", "BranchSnag", 1 - rates["BranchSnag"]],
        ["BranchSnag", "AboveGroundFastSoil", rates["BranchSnag"]],
    ]
    return matrices


def _biomass_turnover(rates: dict[str, np.ndarray]) -> list:

    matrices = [
        [
            "Merch",
            "StemSnag",
            rates["StemAnnualTurnoverRate"],
        ],
        [
            "Foliage",
            "AboveGroundVeryFastSoil",
            rates["FoliageFallRate"],
        ],
        [
            "Other",
            "BranchSnag",
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
    ]
    return matrices


def _dom_decay(
    mean_annual_temp: np.ndarray, parameters: CBMEXNParameters
) -> list:

    dom_pools = [
        "AboveGroundVeryFastSoil",
        "BelowGroundVeryFastSoil",
        "AboveGroundFastSoil",
        "BelowGroundFastSoil",
        "MediumSoil",
        "StemSnag",
        "BranchSnag",
    ]
    dom_pool_flows = {
        "AboveGroundVeryFastSoil": "AboveGroundSlowSoil",
        "BelowGroundVeryFastSoil": "BelowGroundSlowSoil",
        "AboveGroundFastSoil": "AboveGroundSlowSoil",
        "BelowGroundFastSoil": "BelowGroundSlowSoil",
        "MediumSoil": "AboveGroundSlowSoil",
        "StemSnag": "AboveGroundSlowSoil",
        "BranchSnag": "AboveGroundSlowSoil",
    }
    matrix_data = []
    for dom_pool in dom_pools:
        decay_parameter = parameters.get_decay_parameter(dom_pool)
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = cbm_exn_functions.compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["reference_temp"],
            max=decay_parameter["max_rate"],
        )
        matrix_data.append([dom_pool, dom_pool, 1 - decay_rate])
        matrix_data.append(
            [
                dom_pool,
                dom_pool_flows[dom_pool],
                decay_rate * (1 - prop_to_atmosphere),
            ]
        )
        matrix_data.append([dom_pool, "CO2", decay_rate * prop_to_atmosphere])

    return matrix_data


def _slow_decay(
    mean_annual_temp: np.ndarray, parameters: CBMEXNParameters
) -> list:

    matrix_data = []
    for dom_pool in ["AboveGroundSlowSoil", "BelowGroundSlowSoil"]:
        decay_parameter = parameters.get_decay_parameter(dom_pool)
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = cbm_exn_functions.compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["reference_temp"],
            max=decay_parameter["max_rate"],
        )
        matrix_data.append([dom_pool, dom_pool, 1 - decay_rate])
        matrix_data.append(
            [
                dom_pool,
                "CO2",
                decay_rate * prop_to_atmosphere,
            ]
        )

    return matrix_data


def _slow_mixing(rate: float) -> list:
    return [
        ["AboveGroundSlowSoil", "BelowGroundSlowSoil", rate],
        ["AboveGroundSlowSoil", "AboveGroundSlowSoil", 1 - rate],
    ]
