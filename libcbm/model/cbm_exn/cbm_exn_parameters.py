from __future__ import annotations
import os
import json
import pandas as pd


def _load_json(path: str, fn: str):
    with open(os.path.join(path, fn), "r", encoding="utf-8") as fp:
        return json.load(fp)


class CBMEXNParameters:
    """Class for reading and accessing parameters for cbm_exn.
    """
    def __init__(self, path: str):
        """Read a directory containing configuration and parameters for
        cbm_exn

        Args:
            path (str): path to a directory containing cbm_exn parameters
                and configuration

        Raises:
            ValueError: a validation error occurred
        """
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
        """returns the cbm_exn pools as a list of strings

        Returns:
            list[str]: pool names
        """
        return self._pools

    def flux_configuration(self) -> list[dict]:
        """returns cbm_exn's raw flux inidicator json configuration

        Returns:
            list[dict]: flux indicator configuration
        """
        return self._flux

    def get_slow_mixing_rate(self) -> float:
        """gets the CBM slow mixing rate parameter

        Returns:
            float: slow mixing rate
        """
        return self._slow_mixing_rate

    def get_turnover_parameters(self) -> pd.DataFrame:
        """gets a table of turnover parameters used for CBM proportional
        turnovers

        Returns:
            pd.DataFrame: a pandas dataframe of the turnover parameters
        """
        return self._turnover_parameters

    def get_sw_hw_map(self) -> dict[int, int]:
        """returns a map of species identifier to sw_hw
        where the value is either 0: sw or 1: hw

        Returns:
            dict[int, int]: dictionary of species id to 0 (sw) or 1 (hw)
        """
        return self._sw_hw_map

    def get_root_parameters(self) -> dict[str, float]:
        """get the CBM root parameters as a dictionary

        Returns:
            dict[str, float]: named root parameters as a dictionary of
                name: parameter.
        """
        return self._root_parameters

    def get_decay_parameter(self, dom_pool: str) -> dict[str, float]:
        """Get decay parameters for the specified named dead organic matter
        (DOM) pool.

        Args:
            dom_pool (str): the DOM pool

        Returns:
            dict[str, float]: a dictionary of named parameters for that DOM
                pool.
        """
        return self._decay_param_dict[dom_pool]

    def get_disturbance_matrices(self) -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix value information.

        Columns::

         * disturbance_matrix_id
         * source_pool_id
         * sink_pool_id
         * proportion

        Returns:
            pd.DataFrame: a table of disturbance matrix values for CBM
                disturbance C pool flows.
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

        Returns:
            pd.DataFrame: a table of values that associated disturbance matrix
                ids with disturbance type, spatial unit and sw-hw forest type.

        """
        return self._disturbance_matrix_associations
