from enum import IntEnum
import numpy as np
import pandas as pd
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
        self._overmature_decline_op: Operation = None
        self._spinup_net_growth_op: Operation = None
        self._spinup_overmature_decline_op: Operation = None
        self._disturbance_op: Operation = None
        self._dm_associations: pd.DataFrame = None

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

    def disturbance(
        self, disturbance_type: Series, spuid: Series, species: Series
    ) -> Operation:
        if self._disturbance_op is None:
            self._disturbance_op, self._dm_associations = _disturbance(
                self._model, self._parameters
            )
            matrix_idx = self._extract_dm_index(
                disturbance_type, spuid, species
            )

            self._disturbance_op.set_op(matrix_idx)
        else:
            self._disturbance_op.update_index(
                self._extract_dm_index(disturbance_type, spuid, species)
            )

    def _extract_dm_index(
        self, disturbance_type: Series, spuid: Series, species: Series
    ):
        sw_hw = species.map(self._parameters.get_sw_hw_map())
        matrix_index_merge = pd.DataFrame(
            {
                "disturbance_type_id": disturbance_type.to_numpy(),
                "spatial_unit_id": spuid.to_numpy(),
                "sw_hw": sw_hw.to_numpy(),
            }
        ).merge(
            self._dm_associations,
            how="left",
            left_on=["disturbance_type_id", "spatial_unit_id", "sw_hw"],
        )

        # check for cases where the merged idx is null and the disturbance
        # type is >0
        missing_dmidx_loc = pd.isnull(matrix_index_merge["matrix_idx"]) & (
            matrix_index_merge["disturbance_type_id"] > 0
        )
        if missing_dmidx_loc.any():
            missing_rows = matrix_index_merge.loc[missing_dmidx_loc]
            raise ValueError(
                "no disturbance matrices found for the specified "
                "disturbance type/spu/hw_sw combinations: "
                f"{str(missing_rows)}"
            )
        matrix_idx = (
            matrix_index_merge["matrix_idx"].fillna(0).to_numpy(dtype="int32")
        )

        return matrix_idx

    def dom_decay(self, mean_annual_temperature: Series) -> Operation:
        dom_decay_op = _dom_decay(
            self._model, mean_annual_temperature.to_numpy(), self._parameters
        )
        dom_decay_op.set_op(np.arange(0, mean_annual_temperature.length))
        return dom_decay_op

    def slow_decay(self, mean_annual_temperature: Series) -> Operation:
        slow_decay_op = _slow_decay(
            self._model, mean_annual_temperature.to_numpy(), self._parameters
        )
        slow_decay_op.set_op(np.arange(0, mean_annual_temperature.length))
        return slow_decay_op

    def slow_mixing(self, n_rows: int) -> Operation:
        if not self._slow_mixing_op:
            self._slow_mixing_op = _slow_mixing(
                self._model, self._slow_mixing_rate
            )
            self._slow_mixing_op.set_op(np.zeros(n_rows, dtype="int"))
        else:
            self._slow_mixing_op.update_index(np.zeros(n_rows, dtype="int"))
        return self._slow_mixing_op

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

    def net_growth(
        self, cbm_vars: CBMVariables
    ) -> tuple[Operation, Operation]:
        growth_info = cbm_exn_functions.prepare_growth_info(
            cbm_vars, self._parameters
        )
        self._net_growth_op = _net_growth(self._model, growth_info)
        self._overmature_decline_op = _overmature_decline(
            self._model, growth_info
        )
        net_growth = self._net_growth_op.set_op(
            np.arange(0, cbm_vars["pools"].n_rows)
        )
        overmature_decline = self._overmature_decline_op.set_op(
            np.arange(0, cbm_vars["pools"].n_rows)
        )
        return net_growth, overmature_decline

    def _spinup_net_growth_idx(self, spinup_vars: CBMVariables) -> np.ndarray:
        age = spinup_vars["state"]["age"].to_numpy()
        op_index = (
            np.arange(0, self._spinup_2d_shape[0]) * self._spinup_2d_shape[1]
        )
        op_index = np.where(
            age >= self._spinup_2d_shape[1],
            op_index + self._spinup_2d_shape[0] - 1,
            op_index + age,
        )
        return op_index

    def spinup_net_growth(
        self, spinup_vars: CBMVariables
    ) -> tuple[Operation, Operation]:
        if not self._spinup_net_growth_op:
            spinup_growth_info = cbm_exn_functions.prepare_spinup_growth_info(
                spinup_vars, self._parameters
            )

            for k in spinup_growth_info.keys():
                self._spinup_2d_shape = spinup_growth_info[k].shape
                spinup_growth_info[k] = spinup_growth_info[k].flatten(
                    order="C"
                )

            self._spinup_net_growth_op = _net_growth(spinup_growth_info)
            self._spinup_overmature_decline_op = _overmature_decline(
                spinup_growth_info
            )

            op_index = self._spinup_net_growth_idx(spinup_vars)
            self._spinup_net_growth_op.set_op(op_index)
            self._spinup_overmature_decline_op.set_op(op_index)
        else:
            op_index = self._spinup_net_growth_idx(spinup_vars)
            self._spinup_net_growth_op.update_index(op_index)
            self._spinup_overmature_decline_op.update_index(op_index)

        return [self._spinup_net_growth_op, self._spinup_overmature_decline_op]


def _disturbance(
    model: CBMModel, parameters: CBMEXNParameters
) -> tuple[Operation, pd.DataFrame]:
    disturbance_matrices = parameters.get_disturbance_matrices()

    matrix_data = []
    dmid_index = {}
    for idx, dmid in enumerate(
        disturbance_matrices["disturbance_matrix_id"].unique()
    ):
        dmid_loc = disturbance_matrices["disturbance_matrix_id"] == dmid
        dmid_mat_data = []
        for i, row in disturbance_matrices.loc[dmid_loc].iterrows():
            dmid_mat_data.append(
                [row["source"], row["sink"], row["proportion"]]
            )
        matrix_data.append(dmid_mat_data)
        dmid_index[int(dmid)] = idx
    op = model.create_operation(
        matrix_data, fmt="matrix_list", process_id=OpProcesses.disturbance
    )
    dm_association_rows = []
    for _, row in parameters.get_disturbance_matrix_associations().iterrows():
        dm_association_rows.append(
            [
                int(row["disturbance_type_id"]),
                int(row["spatial_unit_id"]),
                int(row["sw_hw"]),
                dmid_index[int(row["disturbance_matrix_id"])],
            ]
        )

    dm_associations = pd.DataFrame(
        columns=[
            "disturbance_type_id",
            "spatial_unit_id",
            "sw_hw",
            "matrix_idx",
        ],
        data=dm_association_rows,
    )
    return (op, dm_associations)


def _net_growth(
    model: CBMModel,
    growth_info: dict[str, np.ndarray],
) -> Operation:
    op = model.create_operation(
        matrices=[
            ["Input", "Merch", growth_info["merch_inc"] * 0.5],
            ["Input", "Other", growth_info["other_inc"] * 0.5],
            ["Input", "Foliage", growth_info["foliage_inc"] * 0.5],
            ["Input", "CoarseRoots", growth_info["coarse_root_inc"] * 0.5],
            ["Input", "FineRoots", growth_info["fine_root_inc"] * 0.5],
        ],
        fmt="repeating_coordinates",
        process_id=OpProcesses.growth,
    )
    return op


def _overmature_decline(
    model: CBMModel,
    growth_info: dict[str, np.ndarray],
) -> Operation:

    op = model.create_operation(
        matrices=[
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
        ],
        process_id=OpProcesses.growth,
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
    model: CBMModel, mean_annual_temp: np.ndarray, parameters: CBMEXNParameters
) -> Operation:

    dom_pools = [
        "AboveGroundVeryFast",
        "BelowGroundVeryFast",
        "AboveGroundFast",
        "BelowGroundFast",
        "MediumSoil",
        "StemSnag",
        "BranchSnag",
    ]
    dom_pool_flows = {
        "AboveGroundVeryFast": "AboveGroundSlow",
        "BelowGroundVeryFast": "BelowGroundSlow",
        "AboveGroundFast": "AboveGroundSlow",
        "BelowGroundFast": "BelowGroundSlow",
        "MediumSoil": "AboveGroundSlow",
        "StemSnag": "AboveGroundSlow",
        "BranchSnag": "AboveGroundSlow",
    }
    matrix_data = []
    for dom_pool in dom_pools:
        decay_parameter = parameters.get_decay_parameter(dom_pool)
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = cbm_exn_functions.compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["tref"],
            max=decay_parameter["max"],
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
    op = model.create_operation(
        matrix_data, "repeating_coordinates", OpProcesses.decay
    )
    return op


def _slow_decay(
    model: CBMModel, mean_annual_temp: np.ndarray, parameters: CBMEXNParameters
) -> Operation:

    matrix_data = []
    for dom_pool in ["AboveGroundSlow", "BelowGroundSlow"]:
        decay_parameter = parameters.get_decay_parameter(dom_pool)
        prop_to_atmosphere = decay_parameter["prop_to_atmosphere"]
        decay_rate = cbm_exn_functions.compute_decay_rate(
            mean_annual_temp=mean_annual_temp,
            base_decay_rate=decay_parameter["base_decay_rate"],
            q10=decay_parameter["q10"],
            tref=decay_parameter["tref"],
            max=decay_parameter["max"],
        )
        matrix_data.append([dom_pool, dom_pool, 1 - decay_rate])
        matrix_data.append(
            [
                dom_pool,
                "CO2",
                decay_rate * prop_to_atmosphere,
            ]
        )

    op = model.create_operation(
        matrix_data, "repeating_coordinates", OpProcesses.decay
    )
    return op


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
