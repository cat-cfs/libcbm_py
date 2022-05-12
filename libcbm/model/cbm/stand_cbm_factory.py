from typing import Union
from typing import Callable
from typing import ContextManager
from typing import Tuple
import pandas as pd
from libcbm.model.cbm.cbm_model import CBM
from libcbm.model.cbm import cbm_defaults
from libcbm.storage.series import Series
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe_functions
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm import resources


def _safe_map(series: Series, map: Union[dict, Callable]):
    """Helper method to ensure an error is thrown if any value in a
    mapped series is null

    Args:
        series (Series): the series to map
        map (Union[dict, Callable]): the map values or map function

    Raises:
        ValueError: at least one value in the mapped series was null

    Returns:
        Series: The mapped series
    """

    out_series = series.map(map)
    null_values = dataframe_functions.is_null(out_series)
    if null_values.any():
        missing_entries = list(series.filter(null_values).unique().to_numpy())
        raise ValueError(f"undefined values detected {missing_entries[:10]}")
    return out_series


class StandCBMFactory:
    """StandCBMFactory encompasses a method to create a working CBM instance
    with minimal inputs: stand merchantable volumes and classifiers.

    The resulting CBM instance can easily be used for stand level simulations
    with explicit relationships between stands and disturbances.
    """

    def __init__(
        self,
        classifiers: dict[str, list],
        merch_volumes: list[dict],
        db_path: str = None,
        locale: str = "en-CA",
        dll_path: str = None,
    ):
        """Initialize an instance of CBMStandFactory using classifiers and
        merch volumes.

        Example classifiers::

            {
                "c1": ["c1_v1", "c1_v2", ...],
                "c1": ["c1_v1", "c1_v2", ...],
                ...
                "cN": ["cN_v1", "cN_v2", ...],
            }

        Example merch_volumes::

                [
                    {
                        "classifier_set": ["c1_v1", "c2_v1", ..., "cN_vK"],
                        "merch_volumes": [
                            "species_name": "Spruce",
                            "age_volume_pairs": [
                                [0, 0],
                                [50, 100],
                                [100, 150],
                                [150, 200],
                            ]
                        ]
                    }
                ]


        Args:
            classifiers (dict): dictionary describing classifiers and
                classifier values
            merch_volumes (list): list of dictionaries describing merchantable
                volume components. See example.
            db_path (str, optional): path to a cbm_defaults database. If None,
                the default database is used. Defaults to None.
            locale_code (str, optional): a locale code used to fetch the
                corresponding translated version of default parameter strings
            dll_path (str, optional): path to the libcbm compiled library, if
                not specified a default value is used.
        """
        if not db_path:
            self._db_path = resources.get_cbm_defaults_path()
        else:
            self._db_path = db_path

        if not dll_path:
            self._dll_path = resources.get_libcbm_bin_path()
        else:
            self._dll_path = dll_path

        self._classifiers = classifiers
        self._merch_volumes = merch_volumes
        self._locale = locale
        self.defaults_ref = CBMDefaultsReference(self._db_path, self._locale)

        self._classifier_config = self._get_classifier_config()
        self._classifier_idx = cbm_config.get_classifier_indexes(
            self._classifier_config
        )
        self.merch_vol_factory = self.merch_volumes_factory()

    def _has_undefined_classifier_values(self, classifier_set):
        """returns true if any non-wildcard value in the classifier_set does
        not correspond to a value in self._classifiers.

        Used as a convenience for when the yield table passed to this class
        contains "extra" values that can not possibly be used due to no linkage
        to any defined inventory/classifier set

        Args:
            classifier_set (list): list of classifier value strings
        """
        classifier_index = self._classifier_idx["classifier_value_ids"]
        for i, c in enumerate(classifier_index.keys()):
            if (
                classifier_set[i] not in classifier_index[c]
                and classifier_set[i] != "?"
            ):
                return True
        return False

    def merch_volumes_factory(self) -> dict:
        merch_volume_list = []
        for c in self._merch_volumes:
            if self._has_undefined_classifier_values(c["classifier_set"]):
                continue
            merch_volume_list.append(
                cbm_config.merch_volume_curve(
                    classifier_set=c["classifier_set"],
                    merch_volumes=[
                        {
                            "species_id": self.defaults_ref.get_species_id(
                                m["species"]
                            ),
                            "age_volume_pairs": [
                                [int(age_vol[0]), float(age_vol[1])]
                                for age_vol in m["age_volume_pairs"]
                            ],
                        }
                        for m in c["merch_volumes"]
                    ],
                )
            )
        return cbm_config.merch_volume_to_biomass_config(
            db_path=self._db_path, merch_volume_curves=merch_volume_list
        )

    def get_disturbance_type_map(self) -> dict:
        return {
            r["disturbance_type_id"]: r["disturbance_type_name"]
            for r in self.defaults_ref.disturbance_type_ref
        }

    def get_classifier_map(self) -> dict:
        return self._classifier_idx["classifier_value_names"].copy()

    def _get_classifier_value_ids(
        self, classifier_name: str, classifier_value_name_series: Series
    ) -> Series:
        classifier_value_name_map = self._classifier_idx[
            "classifier_value_ids"
        ][classifier_name]
        return _safe_map(
            classifier_value_name_series, classifier_value_name_map
        )

    def _get_classifier_config(self):
        classifiers_list = []
        for classifier_name, values in self._classifiers.items():
            classifiers_list.append(
                cbm_config.classifier(
                    classifier_name,
                    values=[
                        cbm_config.classifier_value(value) for value in values
                    ],
                )
            )

        return cbm_config.classifier_config(classifiers_list)

    def classifiers_factory(self):
        return self._classifier_config

    def prepare_inventory(
        self, inventory_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Prepare inventory, classifiers pd.DataFrames compatible with
        :py:func:`libcbm.model.cbm.cbm_simulator.simulate` using the provided
        inventory dataframe.

        Args:
            inventory_df (DataFrame): dataframe with the following columns::

                * c1 .. cN: one column for each classifier, each containing
                  classifier values
                * admin_boundary: the admin boundary name for drawing CBM
                  parameters (defined in db)
                * eco_boundary: the eco boundary name for drawing CBM
                  parameters (defined in db)
                * age: inventory age [years]
                * area: inventory area [hectares]
                * delay: inventory spinup delay [years]
                * land_class: unfccc land class name (defined in db.landclass)
                * afforestation_pre_type: afforestation pre-type name (defined
                  in db)
                * historic_disturbance_type: historic disturbance type name
                  (defined in db)
                * last_pass_disturbance_type: last pass disturbance type name
                  (defined in db)

        Returns:
            Tuple:
                0: classifiers DataFrame
                1: inventory DataFrame
        """

        classifiers = DataFrame(
            columns=self._classifiers.keys(),
            data={
                k: self._get_classifier_value_ids(k, inventory_df[k])
                for k in self._classifiers.keys()
            },
        )
        inventory = DataFrame(
            data={
                "age": inventory_df["age"],
                "area": inventory_df["area"],
                "spatial_unit": _safe_map(
                    inventory_df.index,
                    lambda x: self.defaults_ref.get_spatial_unit_id(
                        str(inventory_df.admin_boundary.loc[x]),
                        str(inventory_df.eco_boundary.loc[x]),
                    ),
                ),
                "afforestation_pre_type_id": _safe_map(
                    inventory_df["afforestation_pre_type"],
                    lambda _, value: (
                        -1
                        if dataframe_functions.is_null(value)
                        else self.defaults_ref.get_afforestation_pre_type_id(
                            value
                        )
                    ),
                ),
                "land_class": _safe_map(
                    inventory_df["land_class"],
                    self.defaults_ref.get_land_class_id,
                ),
                "historical_disturbance_type": _safe_map(
                    inventory_df["historic_disturbance_type"],
                    lambda _, value: (
                        -1
                        if dataframe_functions.is_null(value)
                        else self.defaults_ref.get_disturbance_type_id(value)
                    ),
                ),
                "last_pass_disturbance_type": _safe_map(
                    inventory_df.last_pass_disturbance_type,
                    lambda x: (
                        -1
                        if pd.isnull(x)
                        else self.defaults_ref.get_disturbance_type_id(x)
                    ),
                ),
                "delay": inventory_df["delay"],
            },
            back_end=inventory_df.backend_type,
            nrows=inventory_df.n_rows,
        )
        return classifiers, inventory

    def initialize_cbm(self) -> ContextManager[CBM]:
        return cbm_factory.create(
            dll_path=self._dll_path,
            dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(
                self._db_path
            ),
            cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(
                self._db_path
            ),
            merch_volume_to_biomass_factory=self.merch_volumes_factory,
            classifiers_factory=self.classifiers_factory,
        )
