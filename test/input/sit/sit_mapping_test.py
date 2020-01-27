import unittest
import pandas as pd
from mock import Mock
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.input.sit.sit_cbm_defaults import SITCBMDefaults


class SITMappingTest(unittest.TestCase):

    def get_mock_classifiers(self):
        classifiers = pd.DataFrame(
            data=[
                (1, "classifier1"),
                (2, "classifier2")
            ],
            columns=["id", "name"]
        )
        classifier_values = pd.DataFrame(
            data=[
                (1, "a", "a"),
                (1, "b", "b"),
                (2, "a", "a")
            ],
            columns=["classifier_id", "name", "description"]
        )
        return classifiers, classifier_values

    def test_undefined_default_species_error(self):
        """Checks that an error is raised when the default mapping of species
        does not match a defined value in the defaults reference.
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    {"user_species": "a", "default_species": "Spruce"},
                    {"user_species": "b", "default_species": "Oak"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_species_id(species_name):
            # simulates a key error raised when the specified value is not
            # present.
            raise KeyError()
        species = pd.Series(["a", "b"])
        ref.get_species_id.side_effect = mock_get_species_id
        ref.get_afforestation_pre_types.side_effect = lambda: []
        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_species(
                species, classifiers, classifier_values)
        self.assertTrue(ref.get_species_id.called)

    def test_get_species_expected_result(self):
        """checks the expected output of get_species
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    {"user_species": "a", "default_species": "Spruce"},
                    {"user_species": "b", "default_species": "Oak"},
                    {"user_species": "nonforest",
                     "default_species": "Gleysolic"},
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers = pd.DataFrame(
            data=[
                (1, "classifier1"),
                (2, "classifier2")
            ],
            columns=["id", "name"]
        )

        classifier_values = pd.DataFrame(
            data=[
                (1, "a", "a"),
                (1, "b", "b"),
                (1, "nonforest", "nonforest"),
                (2, "a", "a")
            ],
            columns=["classifier_id", "name", "description"]
        )

        def mock_get_species_id(species_name):
            if species_name == "Spruce":
                return 999
            if species_name == "Oak":
                return -999
            raise ValueError()

        species = pd.Series(["a", "b", "a", "b"])
        ref.get_species_id.side_effect = mock_get_species_id
        ref.get_afforestation_pre_types.side_effect = lambda: [
            {"afforestation_pre_type_name": "Gleysolic"}]
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_species(
            species, classifiers, classifier_values)
        self.assertTrue(list(result) == [999, -999, 999, -999])

    def test_get_species_error_on_undefined_classifier(self):
        """checks that an error is thrown if an undefined species classifier
        is used
        """
        config = {
            "species": {
                "species_classifier": "undefined",
                "species_mapping": [
                    {"user_species": "a", "default_species": "Spruce"},
                    {"user_species": "b", "default_species": "Oak"},
                    {"user_species": "nonforest",
                     "default_species": "Gleysolic"},
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        species = pd.Series(["a", "b", "a", "b"])
        ref.get_afforestation_pre_types.side_effect = lambda: [
            {"afforestation_pre_type_name": "Gleysolic"}]
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(ValueError):
            sit_mapping.get_species(
                species, classifiers, classifier_values)

    def test_undefined_default_nonforest_type_error(self):
        """Checks that an error is raised when the default mapping of
        non-forest type does not match a defined value in the defaults
        reference.
        """
        config = {
            "nonforest":
            {
                "nonforest_classifier": "classifier1",
                "nonforest_mapping": [
                    {"user_nonforest_type": "a",
                     "default_nonforest_type": "missing"},
                    {"user_nonforest_type": "b",
                     "default_nonforest_type": None}
                ]
            },
            "species": {
                "species_classifier": "classifier2",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Spruce"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": "1"}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = None
        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_nonforest_cover_ids(
                inventory, classifiers, classifier_values)

    def test_undefined_single_default_spatial_unit_error(self):
        """Checks that an error is raised when the default mapping of spatial
        unit does not match a defined value in the defaults reference.
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SingleDefaultSpatialUnit",
                "default_spuid": 10
            }
        }
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_spatial_unit(spatial_unit_id):
            raise KeyError

        ref.get_spatial_unit.side_effect = mock_get_spatial_unit
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })
        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)
        self.assertTrue(ref.get_spatial_unit.called)

    def test_single_default_spatial_unit_admin_eco_specified(self):
        """Checks that an error is raised when the default mapping of spatial
        unit does not match a defined value in the defaults reference.
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SingleDefaultSpatialUnit",
                "admin_boundary": "a1",
                "eco_boundary": "e1"
            }
        }
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_spatial_unit_id(admin_boundary, eco_boundary):
            self.assertTrue(admin_boundary == "a1")
            self.assertTrue(eco_boundary == "e1")
            return 1

        ref.get_spatial_unit_id.side_effect = mock_get_spatial_unit_id
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })
        sit_mapping = SITMapping(mapping, ref)
        result = sit_mapping.get_spatial_unit(
            inventory, classifiers, classifier_values)
        self.assertTrue((result == 1).all())
        self.assertTrue(len(result) == inventory.shape[0])
        ref.get_spatial_unit_id.assert_called_once()

    def test_undefined_admin_eco_default_spatial_unit_error(self):
        """Checks that an error is raised when the default mapping of spatial
        unit does not match a defined value in the defaults reference in
        separate admin/eco mode
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SeparateAdminEcoClassifiers",
                "admin_classifier": "classifier1",
                "eco_classifier": "classifier2",
                "admin_mapping": [
                    {"user_admin_boundary": "a",
                     "default_admin_boundary": "British Columbia"},
                    {"user_admin_boundary": "b",
                     "default_admin_boundary": "Alberta"}
                ],
                "eco_mapping": [
                    {"user_eco_boundary": "a",
                     "default_eco_boundary": "Montane Cordillera"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_spatial_unit_id(admin, eco):
            # simulates a key error raised when the specified value is not
            # present.
            raise KeyError()

        ref.get_spatial_unit_id.side_effect = mock_get_spatial_unit_id
        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)
        self.assertTrue(ref.get_spatial_unit_id.called)

    def test_spatial_unit_mapping_returns_expected_value(self):
        """checks that an expected value is returned when spatial unit
        classifier mapping is used"""
        mapping = {
            "spatial_units": {
                "mapping_mode": "JoinedAdminEcoClassifier",
                "spu_classifier": "classifier1",
                "spu_mapping": [
                    {
                        "user_spatial_unit": "a",
                        "default_spatial_unit": {
                            "admin_boundary": "British Columbia",
                            "eco_boundary": "Montane Cordillera"
                        }
                    },
                    {
                        "user_spatial_unit": "b",
                        "default_spatial_unit": {
                            "admin_boundary": "Alberta",
                            "eco_boundary": "Montane Cordillera"
                        }
                    }
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_spatial_unit_id(admin, eco):
            if admin == "British Columbia" and eco == "Montane Cordillera":
                return 1000
            elif admin == "Alberta" and eco == "Montane Cordillera":
                return 2000
            else:
                raise ValueError

        ref.get_spatial_unit_id.side_effect = mock_get_spatial_unit_id
        sit_mapping = SITMapping(mapping, ref)
        result = sit_mapping.get_spatial_unit(
            inventory, classifiers, classifier_values)
        self.assertTrue(list(result) == [1000, 2000])

    def test_admin_eco_mapping_returns_expected_value(self):
        """Checks that an expected value is returned when admin-eco
        classifier mapping is used.
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SeparateAdminEcoClassifiers",
                "admin_classifier": "classifier1",
                "eco_classifier": "classifier2",
                "admin_mapping": [
                    {"user_admin_boundary": "a",
                     "default_admin_boundary": "British Columbia"},
                    {"user_admin_boundary": "b",
                     "default_admin_boundary": "Alberta"}
                ],
                "eco_mapping": [
                    {"user_eco_boundary": "a",
                     "default_eco_boundary": "Montane Cordillera"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_spatial_unit_id(admin, eco):
            if admin == "British Columbia" and eco == "Montane Cordillera":
                return 1000
            elif admin == "Alberta" and eco == "Montane Cordillera":
                return 2000
            else:
                raise ValueError

        ref.get_spatial_unit_id.side_effect = mock_get_spatial_unit_id
        sit_mapping = SITMapping(mapping, ref)
        result = sit_mapping.get_spatial_unit(
            inventory, classifiers, classifier_values)
        self.assertTrue(list(result) == [1000, 2000])

    def test_undefined_mapped_default_spatial_unit_error(self):
        """Checks that an error is raised when the default mapping of spatial
        unit does not match a defined value in the defaults reference in
        spu classifier mode
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "JoinedAdminEcoClassifier",
                "spu_classifier": "classifier1",
                "spu_mapping": [
                    {
                        "user_spatial_unit": "a",
                        "default_spatial_unit": {
                            "admin_boundary": "British Columbia",
                            "eco_boundary": "Montane Cordillera"
                        }
                    },
                    {
                        "user_spatial_unit": "b",
                        "default_spatial_unit": {
                            "admin_boundary": "Alberta",
                            "eco_boundary": "Montane Cordillera"
                        }
                    }
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_spatial_unit_id(admin, eco):
            # simulates a key error raised when the specified value is not
            # present.
            raise KeyError

        ref.get_spatial_unit_id.side_effect = mock_get_spatial_unit_id
        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)
        self.assertTrue(ref.get_spatial_unit_id.called)

    def test_undefined_classifier_separate_admin_eco_error(self):
        """checks that an error is raised when any mapping references a
        non-existant classifier name
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SeparateAdminEcoClassifiers",
                "admin_classifier": "NOT DEFINED",
                "eco_classifier": "classifier2",
                "admin_mapping": [
                    {"user_admin_boundary": "a",
                     "default_admin_boundary": "British Columbia"},
                    {"user_admin_boundary": "b",
                     "default_admin_boundary": "Alberta"}
                ],
                "eco_mapping": [
                    {"user_eco_boundary": "a",
                     "default_eco_boundary": "Montane Cordillera"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_undefined_classifier_joined_admin_eco_error(self):
        """checks that an error is raised when any mapping references a
        non-existant classifier name
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "JoinedAdminEcoClassifier",
                "spu_classifier": "NOT_DEFINED",
                "spu_mapping": [
                    {
                        "user_spatial_unit": "a",
                        "default_spatial_unit": {
                            "admin_boundary": "British Columbia",
                            "eco_boundary": "Montane Cordillera"
                        }
                    },
                    {
                        "user_spatial_unit": "b",
                        "default_spatial_unit": {
                            "admin_boundary": "Alberta",
                            "eco_boundary": "Montane Cordillera"
                        }
                    }
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(mapping, ref)
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_undefined_user_admin_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of admin mapping
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SeparateAdminEcoClassifiers",
                "admin_classifier": "classifier1",
                "eco_classifier": "classifier2",
                "admin_mapping": [
                    {"user_admin_boundary": "missing",
                     "default_admin_boundary": "Alberta"}
                ],
                "eco_mapping": [
                    {"user_eco_boundary": "a",
                     "default_eco_boundary": "Montane Cordillera"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_undefined_user_eco_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of eco mapping
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "SeparateAdminEcoClassifiers",
                "admin_classifier": "classifier1",
                "eco_classifier": "classifier2",
                "admin_mapping": [
                    {"user_admin_boundary": "a",
                     "default_admin_boundary": "British Columbia"},
                    {"user_admin_boundary": "b",
                     "default_admin_boundary": "Alberta"}
                ],
                "eco_mapping": [
                    {"user_eco_boundary": "missing",
                     "default_eco_boundary": "Montane Cordillera"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_undefined_user_spatial_unit_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of spatial unit mapping
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "JoinedAdminEcoClassifier",
                "spu_classifier": "classifier1",
                "spu_mapping": [
                    {
                        "user_spatial_unit": "MISSING",
                        "default_spatial_unit": {
                            "admin_boundary": "British Columbia",
                            "eco_boundary": "Montane Cordillera"
                        }
                    },
                    {
                        "user_spatial_unit": "b",
                        "default_spatial_unit": {
                            "admin_boundary": "Alberta",
                            "eco_boundary": "Montane Cordillera"
                        }
                    }
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_undefined_user_species_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of species mapping
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    {"user_species": "UNDEFINED", "default_species": "Spruce"},
                    {"user_species": "b", "default_species": "Oak"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        ref.get_afforestation_pre_types = lambda: []
        species = pd.Series(["b", "b"])
        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_species(
                species, classifiers, classifier_values)

    def test_duplicate_user_species_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of species mapping
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    # note "a" is specified more than one time here
                    {"user_species": "a", "default_species": "Spruce"},
                    {"user_species": "a", "default_species": "Oak"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        ref.get_afforestation_pre_types = lambda: []
        species = pd.Series(["a", "a"])
        with self.assertRaises(ValueError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_species(
                species, classifiers, classifier_values)

    def test_undefined_user_nonforest_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of nonforest mapping
        """
        config = {
            "nonforest":
            {
                "nonforest_classifier": "classifier1",
                "nonforest_mapping": [
                    {"user_nonforest_type": "b",
                     "default_nonforest_type": None}
                ]
            },
            "species": {
                "species_classifier": "classifier2",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Spruce"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": "1"}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = None
        with self.assertRaises(ValueError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_nonforest_cover_ids(
                inventory, classifiers, classifier_values)

    def test_undefined_user_disturbance_type_error(self):
        """checks that an error is raised when a disturbance type is
        not present in the user value of disturbance type mapping
        """
        config = {
            "disturbance_types": [
                {"user_dist_type": "fire", "default_dist_type": "Wildfire"}
            ]
        }
        ref = Mock(spec=SITCBMDefaults)
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_default_disturbance_type_id(
                pd.Series(["missing_value"]))

    def test_undefined_default_disturbance_type_error(self):
        """checks that an error is raised when a default disturbance
        type is not present in the disturbance type reference
        """
        config = {
            "disturbance_types": [
                {"user_dist_type": "fire", "default_dist_type": "Wildfire"}
            ]
        }
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_disturbance_type_id(name):
            raise KeyError()

        ref.get_default_disturbance_type_id.side_effect = \
            mock_get_disturbance_type_id
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_default_disturbance_type_id(
                pd.Series(["fire"]))
        self.assertTrue(ref.get_default_disturbance_type_id.called)

    def test_duplicated_user_disturbance_type_error(self):
        """checks that an error is raised when a disturbance type is
        duplicated in the sit json config
        """
        config = {
            "disturbance_types": [
                {"user_dist_type": "duplicated",
                 "default_dist_type": "Wildfire"},
                {"user_dist_type": "duplicated",
                 "default_dist_type": "Clearcut"}
            ]
        }
        ref = Mock(spec=SITCBMDefaults)
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(KeyError):
            sit_mapping.get_default_disturbance_type_id(
                pd.Series(["duplicated"]))

    def test_get_disturbance_type_id_returns_expected_value(self):
        """Checks the expected output of SITMapping.get_disturbance_type_id
        """
        config = {
            "disturbance_types": [
                {"user_dist_type": "fire",
                 "default_dist_type": "Wildfire"},
                {"user_dist_type": "clearcut",
                 "default_dist_type": "ClearCut"}
            ]
        }
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_disturbance_type_id(name):
            if name == "Wildfire":
                return 1
            if name == "ClearCut":
                return 2
            raise ValueError()

        ref.get_default_disturbance_type_id.side_effect = mock_get_disturbance_type_id
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_default_disturbance_type_id(
            pd.Series(["fire"]+["clearcut"]))
        self.assertTrue(list(result) == [1, 2])

    def test_invalid_spatial_unit_mapping_mode_error(self):
        """checks that a non-supported mapping mode results in error
        """
        mapping = {
            "spatial_units": {
                "mapping_mode": "UNSUPPORTED",
                "default_spuid": 10
            }
        }
        ref = Mock(spec=SITCBMDefaults)

        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })
        sit_mapping = SITMapping(mapping, ref)
        with self.assertRaises(ValueError):
            sit_mapping.get_spatial_unit(
                inventory, classifiers, classifier_values)

    def test_nonforest_classifier_and_species_nonforest_error(self):
        """checks that an error is raised when a non-forest classifier is
        defined, and non-forest values also appear in the species classifier.
        """
        config = {
            "nonforest":
            {
                "nonforest_classifier": "classifier1",
                "nonforest_mapping": [
                    {"user_nonforest_type": "a",
                     "default_nonforest_type": "Gleysolic"},
                    {"user_nonforest_type": "b",
                     "default_nonforest_type": None}
                ]
            },
            "species": {
                "species_classifier": "classifier2",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Gleysolic"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": "1"}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = lambda: []
        with self.assertRaises(ValueError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_nonforest_cover_ids(
                inventory, classifiers, classifier_values)

    def test_same_nonforest_classifier_and_species_classifier_error(self):
        """checks that an error is raised when a the the species classifier
        maps to at least one non forest value and a non-forest classifier is
        used
        """
        config = {
            "nonforest":
            {
                "nonforest_classifier": "classifier1",
                "nonforest_mapping": [
                    {"user_nonforest_type": "a",
                     "default_nonforest_type": "Gleysolic"},
                    {"user_nonforest_type": "b",
                     "default_nonforest_type": None}
                ]
            },
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Spruce"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": "1"}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = lambda: []
        with self.assertRaises(ValueError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_nonforest_cover_ids(
                inventory, classifiers, classifier_values)

    def test_expected_result_with_species_nonforest_values(self):
        """checks that an error is raised when a the the species classifier
        maps to at least one non forest value and a non-forest classifier is
        used
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Spruce"},
                    {"user_species": "b",
                     "default_species": "Gleysolic"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b", "a", "b"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": 1001}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = lambda: [
            {"species_name": "Spruce"}
        ]
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_nonforest_cover_ids(
            inventory, classifiers, classifier_values)
        self.assertTrue(list(result) == [-1, 1001, -1, 1001])

    def test_expected_result_with_nonforest_classifier_values(self):
        """checks that an error is raised when a the the species classifier
        maps to at least one non forest value and a non-forest classifier is
        used
        """
        config = {
            "nonforest":
            {
                "nonforest_classifier": "classifier1",
                "nonforest_mapping": [
                    {"user_nonforest_type": "a",
                     "default_nonforest_type": "Gleysolic"},
                    {"user_nonforest_type": "b",
                     "default_nonforest_type": None}
                ]
            },
            "species": {
                "species_classifier": "classifier2",
                "species_mapping": [
                    {"user_species": "a",
                     "default_species": "Spruce"}
                ]
            }
        }
        ref = Mock(spec=SITCBMDefaults)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b", "a", "b"]})

        def mock_get_afforestation_pre_types():
            return [
                {"afforestation_pre_type_name": "Gleysolic",
                 "afforestation_pre_type_id": 15}
            ]
        ref.get_afforestation_pre_types.side_effect = \
            mock_get_afforestation_pre_types
        ref.get_species.side_effect = lambda: [
            {"species_name": "Spruce"}
        ]
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_nonforest_cover_ids(
            inventory, classifiers, classifier_values)
        self.assertTrue(list(result) == [15, -1, 15, -1])

    def test_get_landclass_undefined_code_error(self):
        """checks that an error is raised if an undefined code is used in the
        input
        """
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_land_classes():
            return [
                {"code": "code1", "land_class_id": 100},
                {"code": "code2", "land_class_id": 200}
            ]

        ref.get_land_classes.side_effect = mock_get_land_classes
        config = {}
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(ValueError):
            sit_mapping.get_land_class_id(
                pd.Series(["code_missing"]))

    def test_get_landclass_undefined_id_error(self):
        """checks that an error is raised if an undefined id is used in the
        input"""
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_land_classes():
            return [
                {"code": "code1", "land_class_id": 100},
                {"code": "code2", "land_class_id": 200}
            ]

        ref.get_land_classes.side_effect = mock_get_land_classes
        config = {}
        sit_mapping = SITMapping(config, ref)
        with self.assertRaises(ValueError):
            sit_mapping.get_land_class_id(
                pd.Series([7000]))

    def test_get_landclass_id_expected_value(self):
        """tests the expected return of get_land_class_id when a Series of
        land class id integers are passed"""
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_land_classes():
            return [
                {"code": "code1", "land_class_id": 100},
                {"code": "code2", "land_class_id": 200}
            ]

        ref.get_land_classes.side_effect = mock_get_land_classes
        config = {}
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_land_class_id(
            pd.Series([100, 200, 100]))
        # for this case, a validated copy of the input series is returned
        self.assertTrue(list(result) == [100, 200, 100])

    def test_get_landclass_id_expected_value_with_code(self):
        """tests the expected return of get_land_class_id when a Series of
        land class code strings are passed"""
        ref = Mock(spec=SITCBMDefaults)

        def mock_get_land_classes():
            return [
                {"code": "code1", "land_class_id": 1000},
                {"code": "code2", "land_class_id": 2000}
            ]

        ref.get_land_classes.side_effect = mock_get_land_classes
        config = {}
        sit_mapping = SITMapping(config, ref)
        result = sit_mapping.get_land_class_id(
            pd.Series(["code1", "code2", "code1"]))
        self.assertTrue(list(result) == [1000, 2000, 1000])
