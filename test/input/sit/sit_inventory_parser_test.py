import unittest
import pandas as pd
from libcbm.input.sit import sit_inventory_parser


class SITInventoryParserTest(unittest.TestCase):

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

    def get_mock_disturbance_types(self):
        return pd.DataFrame(
            data=[
                ("dist1", "fire"),
                ("dist2", "clearcut")
            ],
            columns=["id", "name"]
        )

    def test_expected_result_with_using_non_zero_age_class(self):
        """Checks the age class expansion feature "using_age_class"
        """
        inventory_table = pd.DataFrame(
            data=[
                ("b", "a", "TRUE", "1", 1, 0, 0),
                ("a", "a", False, 100, 1, 0, 0),
                ("a", "a", "-1", 4, 1, 0, 0)])

        classifiers, classifier_values = self.get_mock_classifiers()
        age_classes = self.get_mock_age_classes()
        result = sit_inventory_parser.parse(
            inventory_table, classifiers, classifier_values, None, age_classes)
        self.assertTrue(result.shape[0] == len(inventory_table) + 1)
        self.assertTrue(result.area.sum() == 3)
        self.assertTrue(set(result.age) == {100, 4, 1, 2})

    def test_expected_result_with_numeric_classifiers(self):
        """Checks that numeric classifiers that appear in inventory data
        are parsed as strings
        """
        inventory_table = pd.DataFrame(
            data=[
                ("1", 1, "F", "1", 1, 0, 0),
                (2.0, 1, False, 100, 1, 0, 0),
                (1, 1, "-1", 4, 1, 0, 0)])

        classifiers = pd.DataFrame(
            data=[
                (1, "classifier1"),
                (2, "classifier2")
            ],
            columns=["id", "name"]
        )
        classifier_values = pd.DataFrame(
            data=[
                (1, "1", "1"),
                (1, "2.0", "2"),
                (2, "1", "1"),
            ],
            columns=["classifier_id", "name", "description"]
        )

        age_classes = self.get_mock_age_classes()
        result = sit_inventory_parser.parse(
            inventory_table, classifiers, classifier_values, None, age_classes)
        self.assertTrue((result.classifier1 == ["1", "2.0", "1"]).all())

    def test_expected_result_with_using_zeroth_age_class(self):
        """Checks the age class expansion feature "using_age_class"
        """
        inventory_table = pd.DataFrame(
            data=[
                ("b", "a", "TRUE", "0", 1, 0, 0),
                ("a", "a", False, 100, 1, 0, 0),
                ("a", "a", "-1", 4, 1, 0, 0)])

        classifiers, classifier_values = self.get_mock_classifiers()
        age_classes = self.get_mock_age_classes()
        result = sit_inventory_parser.parse(
            inventory_table, classifiers, classifier_values, None,
            age_classes)
        self.assertTrue(result.shape[0] == len(inventory_table))
        self.assertTrue(result.area.sum() == 3)
        self.assertTrue(set(result.age) == {100, 4, 0})

    def test_non_numeric_or_negative_values_raise_exception(self):
        """checks that an error is raised when a negative or non-numeric
        value is used for:
         - area
         - age (when using_age_class is false)
         - delay
         - spatial reference (negatives allowed here)
        """

        classifiers, classifier_values = self.get_mock_classifiers()
        disturbance_types = self.get_mock_disturbance_types()
        for data in [
                ("b", "a", "F", "x", 1, 0, 0),
                ("b", "a", "F", -1, 1, 0, 0),
                ("b", "a", "F", 0, "x", 0, 0),
                ("b", "a", "F", 0, -1, 0, 0),
                ("b", "a", "F", 0, 0, "x", 0),
                ("b", "a", "F", 0, 0, -1, 0),
                ("b", "a", "F", 0, 0, -1, 0, "dist1", "dist1", "x")]:

            inventory_table = pd.DataFrame([data])
            with self.assertRaises(ValueError):
                sit_inventory_parser.parse(
                    inventory_table, classifiers, classifier_values,
                    disturbance_types, None)

    def test_missing_disturbance_type_id_raises_exception(self):
        """checks that an error is raised if a disturbance type id is
        not defined in sit disturbance types metadata
        """
        classifiers, classifier_values = self.get_mock_classifiers()
        disturbance_types = self.get_mock_disturbance_types()
        for data in [("b", "a", "F", 0, 0, 0, 0, "dist1", "MISSING"),
                     ("b", "a", "F", 0, 0, 0, 0, "MISSING", "dist2")]:

            inventory_table = pd.DataFrame([data])
            with self.assertRaises(ValueError):
                sit_inventory_parser.parse(
                    inventory_table, classifiers, classifier_values,
                    disturbance_types, None)

    def test_missing_classifier_values_raise_exception(self):
        """checks that an error is raised when a classifier value is not
        defined in sit classifiers metadata.
        """
        classifiers, classifier_values = self.get_mock_classifiers()
        disturbance_types = self.get_mock_disturbance_types()
        for data in [("MISSING", "a", "F", 0, 0, 0, 0),
                     ("b", "MISSING", "F", 0, 0, 0, 0)]:

            inventory_table = pd.DataFrame([data])
            with self.assertRaises(ValueError):
                sit_inventory_parser.parse(
                    inventory_table, classifiers, classifier_values,
                    disturbance_types, None)

    def test_spatial_reference_mixed_with_using_age_class(self):
        """checks that an error is raised when a row has both non-negative
        spatial reference value and a True "using_age_class" value
        """
        classifiers, classifier_values = self.get_mock_classifiers()
        disturbance_types = self.get_mock_disturbance_types()
        age_classes = self.get_mock_age_classes()
        for data in [("b", "a", True, 1, 0, 0, 0, "dist1", "dist1", 1)]:

            inventory_table = pd.DataFrame([data])
            with self.assertRaises(ValueError):
                sit_inventory_parser.parse(
                    inventory_table, classifiers, classifier_values,
                    disturbance_types, age_classes)

    def test_duplicate_spatial_reference_raises_error(self):
        """Checks that any duplicate spatial reference values in the inventory
        table triggers an error
        """
        classifiers, classifier_values = self.get_mock_classifiers()
        disturbance_types = self.get_mock_disturbance_types()
        age_classes = self.get_mock_age_classes()

        inventory_table = pd.DataFrame([
            ("a", "a", False, 100, 1, 0, 0, "dist2", "dist1", 10000),
            ("a", "a", "-1", 4, 1, 0, 0, "dist1", "dist1", 10000)])
        # note the same identifier 10000 appears 2 times

        with self.assertRaises(ValueError):
            sit_inventory_parser.parse(
                inventory_table, classifiers, classifier_values,
                disturbance_types, age_classes)
