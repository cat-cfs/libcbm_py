from __future__ import annotations
from typing import Callable
from typing import Iterator
from typing import Tuple
from libcbm.model.cbm.cbm_model import CBM
from libcbm.model.cbm import cbm_defaults
from libcbm.storage.series import Series
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
from libcbm.storage import series
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm import resources


def _apply(series: Series, func: Callable) -> Series:
    """Helper method to ensure an error is thrown if any value in a
    mapped series is null

    Args:
        series (Series): the series to map
        func (Callable): map function

    Returns:
        Series: The mapped series
    """
    _map = {x: func(x) for x in series.to_list()}
    try:
        out_series = series.map(_map)
    except KeyError:
        raise KeyError(
            "mapping values failed: "
            f"series values: {series.to_list()[:10]} ...,"
            f"map values: {list(_map.items())[:10]} ..."
        )
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
        non_identifiers = []
        for c in classifiers.keys():
            if not c.isidentifier():
                non_identifiers.append(c)
        if non_identifiers:
            raise ValueError(
                "The following classifier names are not valid python "
                f"identifiers: {non_identifiers}"
            )
        self._merch_volumes = merch_volumes
        self._locale = locale
        self.defaults_ref = CBMDefaultsReference(self._db_path, self._locale)

        self._classifier_config = self._get_classifier_config()
        self._classifier_idx = cbm_config.get_classifier_indexes(
            self._classifier_config
        )
        self.merch_vol_factory = self.merch_volumes_factory()

        self._disturbance_type_map = {
            int(r["disturbance_type_id"]): str(r["disturbance_type_name"])
            for r in self.defaults_ref.disturbance_type_ref
        }
        self._disturbance_type_map[0] = ""

    def _has_undefined_classifier_values(self, classifier_set: list) -> bool:
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

    @property
    def disturbance_types(self) -> dict[int, str]:
        """dictionary of disturbance type id to disturbance type name"""
        return self._disturbance_type_map

    @property
    def classifier_names(self) -> dict[int, str]:
        """dictionary of classifier id to classifier name"""
        return self._classifier_idx["classifier_names"]

    @property
    def classifier_ids(self) -> dict[str, int]:
        """dictionary of classifier name to classifier id"""
        return self._classifier_idx["classifier_ids"]

    @property
    def classifier_value_ids(self) -> dict[str, dict[str, int]]:
        """
        nested dictionary of classifier name (outer key) to classifier value
        name (inner key) to classifier value id.
        """
        return self._classifier_idx["classifier_value_ids"]

    @property
    def classifier_value_names(self) -> dict[int, str]:
        """dictionary of classifier value id to classifier value name"""
        return self._classifier_idx["classifier_value_names"]

    def _get_classifier_value_ids(
        self, classifier_name: str, classifier_value_name_series: Series
    ) -> Series:
        classifier_value_name_map = self._classifier_idx[
            "classifier_value_ids"
        ][classifier_name]
        return classifier_value_name_series.map(classifier_value_name_map)

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

    def classifiers_factory(self) -> dict:
        return self._classifier_config

    def prepare_inventory(
        self,
        inventory_df: DataFrame,
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
                  in db) Use the string "None" where no afforestation pre-type
                  is needed
                * historic_disturbance_type: historic disturbance type name
                  (defined in db) Use the string "None" for the default
                  historic disturbance type
                * last_pass_disturbance_type: last pass disturbance type name
                  (defined in db) Use the string "None" for the default
                  last pass disturbance type

        Returns:
            Tuple:
                0: classifiers DataFrame
                1: inventory DataFrame
        """

        classifiers = dataframe.from_series_dict(
            {
                k: self._get_classifier_value_ids(k, inventory_df[k])
                for k in self._classifiers.keys()
            },
            nrows=inventory_df.n_rows,
            back_end=inventory_df.backend_type,
        )

        inventory = dataframe.from_series_dict(
            {
                "age": inventory_df["age"],
                "area": inventory_df["area"],
                "spatial_unit": _apply(
                    series.range(
                        "spatial_unit",
                        0,
                        inventory_df.n_rows,
                        1,
                        "int",
                        back_end=inventory_df.backend_type,
                    ),
                    lambda x: self.defaults_ref.get_spatial_unit_id(
                        str(inventory_df["admin_boundary"].at(x)),
                        str(inventory_df["eco_boundary"].at(x)),
                    ),
                ),
                "afforestation_pre_type_id": _apply(
                    inventory_df["afforestation_pre_type"],
                    lambda x: (
                        -1
                        if x == "None"
                        else self.defaults_ref.get_afforestation_pre_type_id(x)
                    ),
                ),
                "land_class": _apply(
                    inventory_df["land_class"],
                    self.defaults_ref.get_land_class_id,
                ),
                "historical_disturbance_type": _apply(
                    inventory_df["historic_disturbance_type"],
                    lambda x: (
                        -1
                        if x == "None"
                        else self.defaults_ref.get_disturbance_type_id(x)
                    ),
                ),
                "last_pass_disturbance_type": _apply(
                    inventory_df["last_pass_disturbance_type"],
                    lambda x: (
                        -1
                        if x == "None"
                        else self.defaults_ref.get_disturbance_type_id(x)
                    ),
                ),
                "delay": inventory_df["delay"],
            },
            back_end=inventory_df.backend_type,
            nrows=inventory_df.n_rows,
        )
        return classifiers, inventory

    def initialize_cbm(self) -> Iterator[CBM]:
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
