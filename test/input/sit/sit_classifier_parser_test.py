import unittest
import numpy as np
import pandas as pd
from libcbm.input.sit import sit_classifier_parser


class SITClassifierParserTest(unittest.TestCase):

    def test_duplicate_classifier_name_value_error(self):
        """checks that an error is raised when any 2 classifiers have the
        same name
        """
        sit_classifiers_table = pd.DataFrame(
            data=[
                ("1", "_CLASSIFIER", "SAME_NAME", np.nan, np.nan),
                (1, "a", "a", np.nan, np.nan),
                (1, "b", "b", np.nan, np.nan),
                (1, "agg1", "agg2", "a", "b"),
                (1, "agg2", "agg2", "a", "b"),
                (2, "_CLASSIFIER", "SAME_NAME", np.nan, np.nan),
                (2, "a", "a", np.nan, np.nan),
                (2, "agg1", "agg1", "a", np.nan)])

        with self.assertRaises(ValueError):
            sit_classifier_parser.parse(sit_classifiers_table)

    def test_duplicate_classifier_value_name_value_error(self):
        """checks that an error is raise if any 2 classifier value names are
        identical for a given classifier
        """
        sit_classifiers_table = pd.DataFrame(
            data=[
                ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                (1, "SAME_NAME", "a", np.nan, np.nan),
                (1, "SAME_NAME", "b", np.nan, np.nan),
                (1, "agg1", "agg2", "a", "b"),
                (1, "agg2", "agg2", "a", "b"),
                (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                (2, "a", "a", np.nan, np.nan),
                (2, "agg1", "agg1", "a", np.nan)])
        with self.assertRaises(ValueError):
            sit_classifier_parser.parse(sit_classifiers_table)

    def test_classifier_aggregate_validation_errors(self):
        """checks that the function validates classifier aggregates
        """

        with self.assertRaises(ValueError):
            # no two aggregates can have the same name
            sit_classifier_parser.parse(
                pd.DataFrame(data=[
                    ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                    (1, "a", "a", np.nan, np.nan),
                    (1, "b", "b", np.nan, np.nan),
                    (1, "SAME_NAME", "agg2", "a", "b"),
                    (1, "SAME_NAME", "agg2", "a", "b"),
                    (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                    (2, "a", "a", np.nan, np.nan),
                    (2, "agg1", "agg1", "a", np.nan)]))

        with self.assertRaises(ValueError):
            # error when a value in the aggregate is not a defined classifier
            # value name for the classifier
            sit_classifier_parser.parse(
                pd.DataFrame(data=[
                    ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                    (1, "a", "a", np.nan, np.nan),
                    (1, "b", "b", np.nan, np.nan),
                    (1, "agg1", "agg2", "MISSING", "b"),
                    (1, "agg2", "agg2", "a", "b"),
                    (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                    (2, "a", "a", np.nan, np.nan),
                    (2, "agg1", "agg1", "a", np.nan)]))

        with self.assertRaises(ValueError):
            # error when a value in the aggregate is duplicated
            sit_classifier_parser.parse(
                pd.DataFrame(data=[
                    ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                    (1, "DUPLICATE", "a", np.nan, np.nan),
                    (1, "b", "b", np.nan, np.nan),
                    (1, "agg1", "agg2", "DUPLICATE", "DUPLICATE"),
                    (1, "agg2", "agg2", "a", "b"),
                    (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                    (2, "a", "a", np.nan, np.nan),
                    (2, "agg1", "agg1", "a", np.nan)]))

    def test_multiple_classifier_per_id_column_error(self):
        """checks if an error is raise with multiple classifiers defined for a
        single id
        """

        # in the following data (1, _CLASSIFIER) appears on 2 different rows
        sit_classifiers_table = pd.DataFrame(
            data=[
                ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                (1, "_CLASSIFIER", "a", np.nan, np.nan),
                (1, "b", "b", np.nan, np.nan),
                (1, "agg1", "agg2", "a", "b"),
                (1, "agg2", "agg2", "a", "b"),
                (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                (2, "a", "a", np.nan, np.nan),
                (2, "agg1", "agg1", "a", np.nan)])

        with self.assertRaises(ValueError):
            sit_classifier_parser.parse(sit_classifiers_table)

    def test_expected_result(self):
        """checks if an error is raise with multiple classifiers defined for a
        single id
        """

        sit_classifiers_table = pd.DataFrame(
            data=[
                ("1", "_CLASSIFIER", "classifier1", np.nan, np.nan),
                (1, "a", "a", np.nan, np.nan),
                (1, "b", "b", np.nan, np.nan),
                (1, "agg1", "agg1", "a", "b"),
                (1, "agg2", "agg2", "a", "b"),
                (2, "_CLASSIFIER", "classifier2", np.nan, np.nan),
                (2, "a", "a", np.nan, np.nan),
                (2, "agg1", "agg1", "a", np.nan)])

        classifiers, classifier_values, classifier_aggregates = \
            sit_classifier_parser.parse(sit_classifiers_table)

        self.assertTrue(list(classifiers.id) == [1, 2])
        self.assertTrue(list(classifiers.name) == ["classifier1", "classifier2"])

        self.assertTrue(list(classifier_values.classifier_id) == [1, 1, 2])
        self.assertTrue(list(classifier_values.name) == ["a", "b", "a"])
        self.assertTrue(list(classifier_values.description) == ["a", "b", "a"])

        self.assertTrue(len(classifier_aggregates) == 3)
        self.assertTrue(
            classifier_aggregates[0] == {
                'classifier_id': 1,
                'name': 'agg1',
                'description': 'agg1',
                'classifier_values': ['a', 'b']})

        self.assertTrue(
            classifier_aggregates[1] == {
                'classifier_id': 1,
                'name': 'agg2',
                'description': 'agg2',
                'classifier_values': ['a', 'b']})

        self.assertTrue(
            classifier_aggregates[2] == {
                'classifier_id': 2,
                'name': 'agg1',
                'description': 'agg1',
                'classifier_values': ['a']})


    def test_expected_result_with_numeric_values(self):
        """checks that numeric values are converted to strings
        """

        sit_classifiers_table = pd.DataFrame(
            data=[
                ("1", "_CLASSIFIER", "999", np.nan, np.nan),
                (1, 1.0, "a", np.nan, np.nan),
                (1, "b", "b", np.nan, np.nan),
                (1, 2.0, "agg1", 1.0, "b"),
                (1, "agg2", "agg2", 1.0, "b"),
                (2, "_CLASSIFIER", 700, np.nan, np.nan),
                (2, 5, "a", np.nan, np.nan),
                (2, 6, "agg1", "5", np.nan)])

        classifiers, classifier_values, classifier_aggregates = \
            sit_classifier_parser.parse(sit_classifiers_table)

        self.assertTrue(list(classifiers.id) == [1, 2])
        self.assertTrue(list(classifiers.name) == ["999", "700"])