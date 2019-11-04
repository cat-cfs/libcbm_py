import unittest
import pandas as pd
from libcbm.input.sit import sit_yield_parser


class SITYieldParserTest(unittest.TestCase):

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
                (2, "a", "a"),
            ],
            columns=["classifier_id", "name", "description"]
        )
        return classifiers, classifier_values

    def get_mock_age_classes(self):
        return pd.DataFrame(
            data=[
                ("0", 0, 0, 0),
                ("1", 2, 1, 2)],
            columns=["name", "class_size", "start_year", "end_year"]
        )

    def test_expected_result_with_numeric_classifiers(self):
        """Checks that numeric classifiers that appear in yield data
        are parsed as strings
        """
        age_classes = self.get_mock_age_classes()
        num_age_classes = len(age_classes)
        classifiers = pd.DataFrame(
            data=[
                (1, "classifier1"),
                (2, "classifier2")
            ],
            columns=["id", "name"]
        )
        classifier_values = pd.DataFrame(
            data=[
                (1, "1.0", "a"),
                (1, "2", "b"),
                (2, "a", "a"),
            ],
            columns=["classifier_id", "name", "description"]
        )
        yield_table = pd.DataFrame([
            ["2", "?", "sp1"] + [x*15 for x in range(0, num_age_classes)],
            [1.0, "?", "sp1"] + [x*15 for x in range(0, num_age_classes)]
        ])
        result = sit_yield_parser.parse(
            yield_table, classifiers, classifier_values, age_classes)
        self.assertTrue((result.classifier1 == ["2", "1.0"]).all())

    def test_incorrect_number_of_volumes_error(self):
        """checks that the number of volumes is equal to the number of
        specified age classes
        """
        age_classes = self.get_mock_age_classes()
        num_age_classes = len(age_classes)
        classifiers, classifier_values = self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            yield_table = pd.DataFrame([
                ["a", "?", "sp1"] +
                [x*15 for x in range(0, num_age_classes+1)]])
            sit_yield_parser.parse(
                yield_table, classifiers, classifier_values, age_classes)
        with self.assertRaises(ValueError):
            yield_table = pd.DataFrame([
                ["a", "?", "sp1"] +
                [x*15 for x in range(0, num_age_classes-1)]])
            sit_yield_parser.parse(
                yield_table, classifiers, classifier_values, age_classes)

    def test_incorrect_number_of_classifiers_error(self):
        """checks that the format has the correct number of columns
        according to the defined classifiers
        """
        age_classes = self.get_mock_age_classes()
        num_age_classes = len(age_classes)
        classifiers, classifier_values = self.get_mock_classifiers()
        for c_set in [["a", "sp1"], ["a", "?", "c", "sp1"]]:
            with self.assertRaises(ValueError):
                yield_table = pd.DataFrame([
                    c_set +
                    [x*15 for x in range(0, num_age_classes)]])
                sit_yield_parser.parse(
                    yield_table, classifiers, classifier_values, age_classes)

    def test_undefined_classifier_value_error(self):
        """checks that the format has values that are either wildcards or
        classifier sets drawn from the set of defined classifiers values
        """
        age_classes = self.get_mock_age_classes()
        num_age_classes = len(age_classes)
        classifiers, classifier_values = self.get_mock_classifiers()
        for c_set in [["a", "UNDEFINED", "sp1"], ["UNDEFINED", "?", "sp1"]]:
            with self.assertRaises(ValueError):
                yield_table = pd.DataFrame([
                    c_set +
                    [x*15 for x in range(0, num_age_classes)]])
                sit_yield_parser.parse(
                    yield_table, classifiers, classifier_values, age_classes)

    def test_non_numeric_or_negative_volume_error(self):
        """checks that an error is raised if any volume is non-numeric or
        is a negative number.
        """
        age_classes = self.get_mock_age_classes()
        num_age_classes = len(age_classes)
        classifiers, classifier_values = self.get_mock_classifiers()
        volumes = [
            [x*-15 for x in range(0, num_age_classes)],
            ["invalid" for x in range(0, num_age_classes)]
        ]

        for vol in volumes:
            with self.assertRaises(ValueError):
                yield_table = pd.DataFrame([["a", "?", "sp1"] + vol])
                sit_yield_parser.parse(
                    yield_table, classifiers, classifier_values, age_classes)
