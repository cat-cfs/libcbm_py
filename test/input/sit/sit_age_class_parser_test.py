import unittest
import pandas as pd
from libcbm.input.sit import sit_age_class_parser


class SITAgeClassParserTest(unittest.TestCase):

    def test_non_zero_initial_size_error(self):
        """Check that a non zero size for the first age class row results in
        error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse(
                pd.DataFrame([(0, 1), (1, 1)]))

    def test_zero_size_error(self):
        """Check that a zero size for any but the first age class row results
        in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse(
                pd.DataFrame([(0, 0), (1, 0)]))

    def test_non_numeric_size_error(self):
        """Check that a non numeric value for the size column results in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse(
                pd.DataFrame([(0, 0), (1, "a")]))

    def test_duplicate_id_error(self):
        """Check that a duplicate value for the id column results in error
        """
        with self.assertRaises(ValueError):
            sit_age_class_parser.parse(
                pd.DataFrame([(0, 0), (1, 1), (1, 1)]))

    def test_generate_sit_age_classes(self):
        """checks the output of the test_generate_sit_age_classes helper method
        """
        for class_size, n_classes in [(1, 5), (2, 10), (10, 20)]:
            n = len(range(1, n_classes, class_size))
            result = sit_age_class_parser.generate_sit_age_classes(
                class_size, n_classes)
            self.assertEqual(
                list(result.iloc[:, 0]), list(range(0, n + 1)))
            self.assertEqual(
                list(result.iloc[:, 1]), [0]+[class_size] * n)

    def test_generate_sit_age_classes_zero_or_less_errors(self):
        """checks that errors are raise on invalid input
        """
        for a, b in [(0, 5), (-1, 10), (10, -20), (10, 0)]:
            with self.assertRaises(ValueError):
                sit_age_class_parser.generate_sit_age_classes(a, b)
