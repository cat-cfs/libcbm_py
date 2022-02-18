# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_defaults
from libcbm.input.sit.sit_reader import SITData
from typing import Callable


class SITCBMDefaults(CBMDefaultsReference):
    def __init__(
        self, sit_data: SITData, db_path: str, locale_code: str = "en-CA"
    ):
        super().__init__(db_path, locale_code)
        self.db_path = db_path

        self.sit_disturbance_type_id_lookup = {
            x["id"]: x["sit_disturbance_type_id"]
            for _, x in sit_data.disturbance_types.iterrows()
        }
        self.default_disturbance_id_lookup = {
            x["disturbance_type_name"]: x["disturbance_type_id"]
            for x in self.get_disturbance_types()
        }

    def get_sit_disturbance_type_id(self, sit_dist_type_name: str) -> int:
        if sit_dist_type_name in self.sit_disturbance_type_id_lookup:
            return self.sit_disturbance_type_id_lookup[sit_dist_type_name]
        else:
            raise KeyError(
                f"Specified sit disturbance type value {sit_dist_type_name} "
                "not mapped."
            )

    def get_configuration_factory(self) -> Callable[[], dict]:
        return cbm_defaults.get_libcbm_configuration_factory(self.db_path)

    def _replace_dist_type(self, disturbance_type_map, parameters):
        """Replaces default disturbance types with SIT defined types in CBM
        config.

        Args:
            disturbance_type_map (dict): the dictionary of SIT disturbance
                type (key) to mapped default disturbance type (value)
            parameters (dict): CBM parameter config dictionary
        """
        parameter_names = [
            "disturbance_matrix_associations",
            "land_class_transitions",
        ]
        for parameter_name in parameter_names:
            df = parameters[parameter_name]
            output = pd.DataFrame()
            for k, v in disturbance_type_map.items():
                matching_rows = df.loc[df["disturbance_type_id"] == v].copy()
                matching_rows["disturbance_type_id"] = k
                output = output.append(matching_rows)
            output = output.reset_index(drop=True)
            parameters[parameter_name] = output

    def _get_disturbance_type_map(
        self, sit_disturbance_types: pd.DataFrame
    ) -> dict[int, int]:
        disturbance_type_map = {
            x["sit_disturbance_type_id"]: x["default_disturbance_type_id"]
            for _, x in sit_disturbance_types.iterrows()
        }
        disturbance_type_map.update({0: 0})  # add the null disturbance type
        return disturbance_type_map

    def get_parameters_factory(
        self, sit_disturbance_types: pd.DataFrame
    ) -> Callable[[], dict]:
        param_func = cbm_defaults.get_cbm_parameters_factory(self.db_path)
        default_parameters = param_func()
        disturbance_type_map = self._get_disturbance_type_map(
            sit_disturbance_types
        )
        self._replace_dist_type(disturbance_type_map, default_parameters)

        def factory():
            return default_parameters

        return factory
