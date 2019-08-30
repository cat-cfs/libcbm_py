import unittest
from libcbm.input.sit import sit_yield_parser


class SITYieldParserTest(unittest.TestCase):

    def test_incorrect_number_of_volumes_error(self):
        """checks that the number of volumes is equal to the number of
        specified age classes
        """
        self.fail()

    def test_undefined_leading_species_value_error(self):
        """checks if all values are mapped in the species map parameter
        """
        self.fail()

    def test_incorrect_number_of_classifiers_error(self):
        """checks that the format has the correct number of columns
        according to the defined classifiers
        """
        self.fail()

    def test_undefined_classifier_value_error(self):
        """checks that the format has values that are either wildcards or
        classifier sets drawn from the set of defined classifiers values
        """
        self.fail()

    def test_non_numeric_or_negative_volume_error(self):
        """checks that an error is raised if any volume is non-numeric or
        is a negative number.
        """
        self.fail()
