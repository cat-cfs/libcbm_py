from __future__ import annotations
from typing import Union
import copy
import os
import json
import pandas as pd


CBMEXN_PARAMETERS_DATA = {
    "pools": {"type": list},
    "flux": {"type": list},
    "slow_mixing_rate": {"type": pd.DataFrame},
    "turnover_parameters": {"type": pd.DataFrame},
    "species": {"type": pd.DataFrame},
    "root_parameters": {"type": pd.DataFrame},
    "decay_parameters": {"type": pd.DataFrame},
    "disturbance_matrix_value": {"type": pd.DataFrame},
    "disturbance_matrix_association": {"type": pd.DataFrame},
}


class CBMEXNParameters:
    """Class for storing and pre-processing parameters for cbm_exn."""

    def __init__(self, data: dict[str, Union[list, pd.DataFrame]]):
        """Read a directory containing configuration and parameters for
        cbm_exn

        Args:
            data (CBMEXNParameterData): path to a directory containing cbm_exn
                parameters and configuration

        Raises:
            ValueError: a validation error occurred
        """
        self._data = copy.deepcopy(data)

        self._slow_mixing_rate = float(
            self._data["slow_mixing_rate"].iloc[0, 1]
        )

        if (
            not self._data["turnover_parameters"]["sw_hw"]
            .isin(["sw", "hw"])
            .all()
        ):
            raise ValueError(
                "turnover_parameters.sw_hw values should be one of "
                "'sw' or 'hw'"
            )
        self._data["turnover_parameters"]["sw_hw"] = self._data[
            "turnover_parameters"
        ]["sw_hw"].map({"sw": 0, "hw": 1})

        self._sw_hw_map = {
            int(row["species_id"]): int(0 if row["forest_type_id"] == 1 else 1)
            for _, row in self._data["species"].iterrows()
        }

        rp = self._data["root_parameters"]
        root_param_cols = list(rp.columns)
        self._root_parameters = {
            col: float(rp[col].iloc[0]) for col in root_param_cols[1:]
        }

        decay_params = self._data["decay_parameters"]
        self._decay_param_dict: dict[str, dict[str, float]] = {}
        for _, row in decay_params.iterrows():
            self._decay_param_dict[str(row["pool"])] = {
                col: float(row[col]) for col in decay_params.columns[1:]
            }

        dm_associations = self._data["disturbance_matrix_association"]
        if not dm_associations["sw_hw"].isin(["sw", "hw"]).all():
            raise ValueError(
                "disturbance_matrix_associations.sw_hw values should be one "
                "of sw' or 'hw'"
            )
        dm_associations["sw_hw"] = dm_associations["sw_hw"].map(
            {"sw": 0, "hw": 1}
        )

    def pool_configuration(self) -> list[str]:
        """returns the cbm_exn pools as a list of strings

        Returns:
            list[str]: pool names
        """
        return self._data["pools"]

    def flux_configuration(self) -> list[dict]:
        """returns cbm_exn's raw flux inidicator json configuration

        Returns:
            list[dict]: flux indicator configuration
        """
        return self._data["flux"]

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
        return self._data["turnover_parameters"]

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
        return self._data["disturbance_matrix_value"]

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
        return self._data["disturbance_matrix_association"]


def _load_data_item(dir: str, item_name: str) -> Union[pd.DataFrame, list]:
    item_type = CBMEXN_PARAMETERS_DATA[item_name]["type"]
    if item_type is pd.DataFrame:
        return pd.read_csv(os.path.join(dir, f"{item_name}.csv"))
    elif item_type is list:
        with open(
            os.path.join(dir, f"{item_name}.json"), "r", encoding="utf-8"
        ) as fp:
            return json.load(fp)
    else:
        raise ValueError(f"unsupported type {item_type}")


def parameters_factory(dir: str = None, data: dict = {}) -> CBMEXNParameters:
    if data:
        _data = data.copy()
        # if any keys in CBMEXN_PARAMETERS_DATA are not present in the
        # specified data attempt to fill them from a specified dir
        missing_subset = set(CBMEXN_PARAMETERS_DATA.keys()).difference(
            _data.keys()
        )
        if missing_subset and not dir:
            raise ValueError(
                "the following required data fields are not present in the "
                f"specified data dictionary: {missing_subset}, and no "
                "alternate directory to fetch them was specified."
            )
        for item_name in missing_subset:
            _data[item_name] = _load_data_item(dir, item_name)
        return CBMEXNParameters(_data)
    elif dir:
        _data = {}
        for item_name in CBMEXN_PARAMETERS_DATA.keys():
            _data[item_name] = _load_data_item(dir, item_name)
    else:
        raise ValueError(
            "neither a data dictionary, nor a directory are specified"
        )
    return CBMEXNParameters(_data)
