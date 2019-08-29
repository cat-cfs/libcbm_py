import unittest
from libcbm.input.sit import sit_classifier_parser


class SITClassifierParserTest(unittest.TestCase):

    def test_duplicate_classifier_name_value_error(self):
        """checks that an error is raised when any 2 classifiers have the
        same name
        """
        self.assertTrue(False)

    def test_duplicate_classifier_value_name_value_error(self):
        """checks that an error is raise if any 2 classifier value names are
        identical for a given classifier
        """
        self.assertTrue(False)

    def test_classifier_aggregate_validation_errors(self):
        """checks that the function validates classifier aggregates
        """
        self.assertTrue(False)

    def test_multiple_classifier_per_id_column_error(self):
        """checks if an error is raise with multiple classifiers defined for a
        single id
        """
        self.assertTrue(False)
