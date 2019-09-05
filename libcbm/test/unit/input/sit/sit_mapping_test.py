import unittest
from libcbm.input.sit import sit_mapping


class SITMappingTest(unittest.TestCase):

    def test_undefined_default_species_error(self):
        """Checks that an error is raised when the default mapping of species
        does not match a defined value in the defaults reference.
        """
        self.fail()

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
        self.fail()

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
        self.fail()

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
