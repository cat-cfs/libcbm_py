import unittest
from libcbm.input.sit import sit_inventory_parser


class SITInventoryParserTest(unittest.TestCase):

    def expected_result_with_using_age_class(self):
        """Checks the age class expansion feature "using_age_class"
        (including the 0th age class)
        """
        self.assertTrue(False)

    def non_numeric_or_negative_values_raise_exception(self):
        """checks that an error is raised when a negative or non-numeric
        value is used for:
         - area
         - age (when using_age_class is false)
         - delay
         - spatial reference (negatives allowed here)
        """
        self.assertTrue(False)

    def missing_disturbance_type_id_raises_exception(self):
        """checks that an error is raised if a disturbance type id is
        not defined in sit disturbance types metadata
        """
        self.assertTrue(False)

    def missing_classifier_values_raise_exception(self):
        """checks that an error is raised when a classifier value is not
        defined in sit classifiers metadata.
        """
        self.assertTrue(False)

    def undefined_land_class_id_raises_exception(self):
        """raised if a value in the "land_class_id" column is not a valid land
        class id
        """
        self.assertTrue(False)

    def spatial_reference_mixed_with_using_age_class(self):
        """checks that an error is raised when a row has both non-negative
        spatial reference value and a True "using_age_class" value
        """
        self.assertTrue(False)
