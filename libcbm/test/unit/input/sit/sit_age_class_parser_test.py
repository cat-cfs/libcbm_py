import unittest
from libcbm.input.sit import sit_age_class_parser


class SITAgeClassParserTest(unittest.TestCase):

    def non_zero_initial_size_error(self):
        """Check that a non zero size for the first age class row results in
        error
        """
        self.assertTrue(False)

    def zero_size_error(self):
        """Check that a zero size for any but the first age class row results
        in error
        """
        self.assertTrue(False)

    def non_numeric_size_error(self):
        """Check that a non numeric value for the size column results in error
        """
        self.assertTrue(False)

    def duplicate_id_error(self):
        """Check that a duplicate value for the id column results in error
        """
        self.assertTrue(False)
