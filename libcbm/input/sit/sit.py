from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm import resources
from libcbm.input.sit.sit_cbm_defaults import SITCBMDefaults
from libcbm.input.sit.sit_mapping import SITMapping


class SIT:
    def __init__(self):
        pass

    def _create_disturbance_type_maps(self):
        """Create maps from the internally defined sequential sit disturbance
        type id to all of:


            * default_disturbance_id_map - maps to the default disturbance
              type id
            * disturbance_id_map - maps to the sit disturbance type input "id"
              field (col 0)
            * disturbance_name_map - maps to the sit disturbance type input
              "name" field (col 1)

        Args:
            sit (object): sit instance as returned by :py:func:`load_sit`
        """
        self._default_disturbance_id_map = {
            row.sit_disturbance_type_id: row.default_disturbance_type_id
            for _, row in self._sit_data.disturbance_types.iterrows()
        }
        self._disturbance_id_map = {
            row.sit_disturbance_type_id: row.id
            for _, row in self._sit_data.disturbance_types.iterrows()
        }
        self._disturbance_name_map = {
            row.sit_disturbance_type_id: row["name"]
            for _, row in self._sit_data.disturbance_types.iterrows()
        }

    def _create_classifier_value_maps(self):
        """Creates dictionaries for fetching internally defined identifiers
        and attaches them to the specified sit object instance. Values can
        then be fetched from the sit instance like the following examples::

            classifier_id = sit.classifier_ids["my_classifier_name"]
            classifier_name = sit.classifier_names[1]
            classifier_value_id = \
                sit.classifier_value_ids["classifier1"]["classifier1_value1"]

        The following fields will be assigned to the specified sit instance:

            * classifier_names - dictionary of id (int, key) to
                name (str, value) for each classifier
            * classifier_ids - dictionary of name (str, value)
                to id (int, key) for each classifier
            * classifier_value_ids - nested dictionary, with one entry per
                classifier name. Each nested dictionary contains classifier
                value name (str, key) to classifier value id (int, value)

                Example::

                    {
                        "classifier_name_1": {
                            "classifier_1_value_name_1": 1,
                            "classifier_1_value_name_2": 2
                        },
                        "classifier_name_2": {
                            "classifier_2_value_name_1": 3,
                            "classifier_2_value_name_2": 4
                        },
                        ...
                    }

            * classifier_value_names - nested dictionary, with one entry per
                classifier id. Each nested dictionary contains classifier value
                name (str, key) to classifier value id (int, value)

                Example::

                    {
                        1: {
                            1: "classifier_1_value_name_1",
                            2: "classifier_1_value_name_2"
                        },
                        2: {
                            3: "classifier_2_value_name_1"
                            4: "classifier_2_value_name_2"
                        },
                        ...
                    }

        Args:
            sit (object): sit instance as returned by :py:func:`load_sit`
        """
        classifiers_config = sit_cbm_factory.get_classifiers(
            self._sit_data.classifiers, self._sit_data.classifier_values
        )
        idx = cbm_config.get_classifier_indexes(classifiers_config)
        self._classifier_names = idx["classifier_names"]
        self._classifier_ids = idx["classifier_ids"]
        self._classifier_value_ids = idx["classifier_value_ids"]
        self._classifier_value_names = idx["classifier_value_names"]

    def _initialize_sit_objects(self, db_path=None, locale_code="en-CA"):
        """Load and attach objects required for the SIT

        Args:
            sit (SIT): an instance of the standard import tool class
            db_path (str, optional): path to a cbm_defaults database. If None, the
                default database is used. Defaults to None.
            locale_code (str, optional): a locale code used to fetch the
                corresponding translated version of default parameter strings
        """
        if not db_path:
            db_path = resources.get_cbm_defaults_path()
        sit_defaults = SITCBMDefaults(self, db_path, locale_code=locale_code)
        self._sit_mapping = SITMapping(
            self._config["mapping_config"], sit_defaults
        )
        self._sit_data.disturbance_types.insert(
            0,
            "default_disturbance_type_id",
            self._sit_mapping.get_default_disturbance_type_id(
                self._sit_data.disturbance_types.name
            ),
        )
        self._db_path = db_path
        self._defaults = sit_defaults
        self._create_classifier_value_maps()
        self._create_disturbance_type_maps()
