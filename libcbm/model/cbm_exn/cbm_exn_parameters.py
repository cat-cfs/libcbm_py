from __future__ import annotations
import os
import json
import pandas as pd


def _load_json(path: str, fn: str):
    with open(os.path.join(path, fn), "r", encoding="utf-8") as fp:
        return json.load(fp)


class CBMEXNParameters:
    def __init__(self, path: str):
        self._path = path
        self._pools: list = _load_json(self._path, "pools.json")
        self._flux: list = _load_json(self._path, "flux.json")
        self._slow_mixing_rate = float(
            pd.read_csv(os.path.join(self._path, "slow_mixing_rate.csv")).iloc[
                0, 1
            ]
        )
        self._turnover_parameters = pd.read_csv(
            os.path.join(self._path, "turnover_parameters.csv")
        )
        if not self._turnover_parameters["sw_hw"].isin(["sw", "hw"]).all():
            raise ValueError(
                "turnover_parameters.sw_hw values should be one of "
                "'sw' or 'hw'"
            )
        self._turnover_parameters["sw_hw"] = self._turnover_parameters[
            "sw_hw"
        ].map({"sw": 0, "hw": 1})
        self._species = pd.read_csv(os.path.join(self._path, "species.csv"))
        self._sw_hw_map = {
            int(row["species_id"]): int(0 if row["forest_type_id"] == 1 else 1)
            for _, row in self._species.iterrows()
        }
        rp = pd.read_csv(os.path.join(self._path, "root_parameters.csv"))
        root_param_cols = list(rp.columns)
        self._root_parameters = {
            col: float(rp[col].iloc[0]) for col in root_param_cols[1:]
        }

        decay_params = pd.read_csv(
            os.path.join(self._path, "decay_parameters.csv")
        )
        self._decay_param_dict: dict[str, dict[str, float]] = {}
        for _, row in decay_params.iterrows():
            self._decay_param_dict[str(row["pool"])] = {
                col: float(row[col]) for col in decay_params.columns[1:]
            }

        self._disturbance_matrix_values = pd.read_csv(
            os.path.join(self._path, "disturbance_matrix_value.csv")
        )
        self._disturbance_matrix_associations = pd.read_csv(
            os.path.join(self._path, "disturbance_matrix_association.csv")
        )
        if (
            not self._disturbance_matrix_associations["sw_hw"]
            .isin(["sw", "hw"])
            .all()
        ):
            raise ValueError(
                "disturbance_matrix_associations.sw_hw values should be one "
                "of sw' or 'hw'"
            )
        self._disturbance_matrix_associations[
            "sw_hw"
        ] = self._disturbance_matrix_associations["sw_hw"].map(
            {"sw": 0, "hw": 1}
        )

    def pool_configuration(self) -> list[str]:
        return self._pools

    def flux_configuration(self) -> list[dict]:
        return self._flux

    def get_slow_mixing_rate(self) -> float:
        return self._slow_mixing_rate

    def get_turnover_parameters(self) -> pd.DataFrame:
        return self._turnover_parameters

    def get_sw_hw_map(self) -> dict[int, int]:
        """
        returns a map of speciesid: sw_hw where sw_hw is either 0: sw or 1: hw
        """
        return self._sw_hw_map

    def get_root_parameters(self) -> dict[str, float]:
        return self._root_parameters

    def get_decay_parameter(self, dom_pool: str) -> dict[str, float]:
        return self._decay_param_dict[dom_pool]

    def get_disturbance_matrices(self) -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix value information.

        Columns::

         * disturbance_matrix_id
         * source_pool_id
         * sink_pool_id
         * proportion

        """
        return self._disturbance_matrix_values

    def get_disturbance_matrix_associations(self) -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix assocation information

        Columns::

         * disturbance_type_id
         * spatial_unit_id
         * sw_hw
         * disturbance_matrix_id

        """
        return self._disturbance_matrix_associations
