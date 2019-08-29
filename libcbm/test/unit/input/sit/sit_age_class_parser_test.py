import unittest
import pandas as pd
from libcbm.input.sit import sit_age_class_parser


class SITAgeClassParserTest(unittest.TestCase):

    def test_non_zero_initial_size_error(self):
        """Check that a non zero size for the first age class row results in
        error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse_age_classes(
                pd.DataFrame([(0, 1), (1, 1)]))

    def test_zero_size_error(self):
        """Check that a zero size for any but the first age class row results
        in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse_age_classes(
                pd.DataFrame([(0, 0), (1, 0)]))

    def test_non_numeric_size_error(self):
        """Check that a non numeric value for the size column results in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse_age_classes(
                pd.DataFrame([(0, 0), (1, "a")]))

    def test_duplicate_id_error(self):
        """Check that a duplicate value for the id column results in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse_age_classes(
                pd.DataFrame([(0, 0), (1, 1), (1, 1)]))
