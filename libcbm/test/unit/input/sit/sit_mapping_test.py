import unittest
import pandas as pd
from mock import Mock, call
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference


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
                    {"user_species": "UNDEFINED", "default_species": "Spruce"},
                    {"user_species": "b", "default_species": "Oak"}
                ]
            }
        }
        ref = Mock(spec=CBMDefaultsReference)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_species_id(species_name):
            # simulates a key error raised when the specified value is not
            # present.
            raise KeyError()

        ref.get_species_id.side_effect = mock_get_species_id
        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(config, ref)
            default_species_map = sit_mapping.get_species_map(
                classifiers, classifier_values)
        self.assertTrue(ref.get_species_id.called)

    def test_undefined_nonforest_type_error(self):
        """Checks that an error is raised when the default mapping of
        non-forest type does not match a defined value in the defaults
        reference.
        """
        self.fail()

    def test_undefined_single_default_spatial_unit_error(self):
        """Checks that an error is raised when the default mapping of spatial
        unit does not match a defined value in the defaults reference.
        """
        self.fail()

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
        ref = Mock(spec=CBMDefaultsReference)
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

    def test_admin_eco_mapping_returns_expected_value(self):
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
        ref = Mock(spec=CBMDefaultsReference)
        classifiers, classifier_values = self.get_mock_classifiers()
        inventory = pd.DataFrame({
            "classifier1": ["a", "b"],
            "classifier2": ["a", "a"],
        })

        def mock_get_spatial_unit_id(admin, eco):
            # simulates a key error raised when the specified value is not
            # present.
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
        self.fail()

    def undefined_classifier_error(self):
        """checks that an error is raised when any mapping references a
        non-existant classifier name
        """
        self.fail()

    def test_undefined_user_admin_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of admin mapping
        """
        self.fail()

    def test_undefined_user_eco_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of eco mapping
        """
        self.fail()

    def test_undefined_user_spatial_unit_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of species mapping
        """
        self.fail()

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
        ref = Mock(spec=CBMDefaultsReference)
        classifiers, classifier_values = self.get_mock_classifiers()

        with self.assertRaises(KeyError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_species_map(
                classifiers, classifier_values)

    def test_duplicate_user_species_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of species mapping
        """
        config = {
            "species": {
                "species_classifier": "classifier1",
                "species_mapping": [
                    #note "a" is specified more than one time here
                    {"user_species": "a", "default_species": "Spruce"},
                    {"user_species": "a", "default_species": "Oak"}
                ]
            }
        }
        ref = Mock(spec=CBMDefaultsReference)
        classifiers, classifier_values = self.get_mock_classifiers()

        with self.assertRaises(ValueError):
            sit_mapping = SITMapping(config, ref)
            sit_mapping.get_species_map(
                classifiers, classifier_values)


    def test_undefined_user_nonforest_error(self):
        """checks that an error is raised when a classifier description is
        not present in the user value of nonforest mapping
        """
        self.fail()

    def test_undefined_user_disturbance_type_error(self):
        """checks that an error is raised when a disturbance type is
        not present in the user value of disturbance type mapping
        """
        self.fail()

    def test_invalid_mapping_mode_error(self):
        """checks that a non-supported mapping mode results in error
        """
        self.fail()
