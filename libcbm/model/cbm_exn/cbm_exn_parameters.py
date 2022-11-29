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

    def get_disturbance_matrices() -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix value information.

        Columns::

         * disturbance_matrix_id
         * source_pool_id
         * sink_pool_id
         * proportion

        """
        # TODO: the 0th matrix should be the identity matrix,
        # representing no-disturbance
        pass

    def get_disturbance_matrix_associations() -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix assocation information

        Columns::

         * disturbance_type_id
         * spatial_unit_id
         * sw_hw
         * disturbance_matrix_id

        """
        # TODO: ensure that no row has a value of zero for
        # disturbance_matrix_id
        pass
