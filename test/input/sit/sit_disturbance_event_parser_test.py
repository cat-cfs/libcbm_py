import unittest
import pandas as pd
from libcbm.input.sit import sit_disturbance_event_parser
from libcbm.input.sit import sit_format


class SITDisturbanceEventParserTest(unittest.TestCase):

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
                (2, "a", "a")
            ],
            columns=["classifier_id", "name", "description"]
        )
        aggregates = [
            {'classifier_id': 1,
             'name': 'agg1',
             'description': 'agg2',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 1,
             'name': 'agg2',
             'description': 'agg2',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 2,
             'name': 'agg1',
             'description': 'agg1',
             'classifier_values': ['a']}]

        return classifiers, classifier_values, aggregates

    def get_mock_age_classes(self):
        return pd.DataFrame(
            data=[
                ("age0", 0, 0, 0),
                ("age1", 2, 1, 2),
                ("age2", 2, 3, 4)],
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

    def assemble_disturbance_events_table(self, events):
        return pd.DataFrame([
            event["classifier_set"] +
            event["age_eligibility"] +
            event["eligibility"] +
            event["target"]
            for event in events
        ])

    def get_num_eligibility_cols(self):
        num_eligibility_cols = len(
            sit_format.get_disturbance_eligibility_columns(0))
        return num_eligibility_cols

    def test_expected_value_with_numeric_classifier_values(self):
        """Checks that numeric classifiers that appear in events data
        are parsed as strings
        """
        event = {
            "classifier_set": [1, 2.0],
            "age_eligibility": ["False", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "dist1", 2, 100]}

        classifiers = pd.DataFrame(
            data=[
                (1, "classifier1"),
                (2, "classifier2")
            ],
            columns=["id", "name"]
        )
        classifier_values = pd.DataFrame(
            data=[
                (1, "1", "a"),
                (1, "b", "b"),
                (2, "a", "a")
            ],
            columns=["classifier_id", "name", "description"]
        )
        aggregates = [
            {'classifier_id': 1,
             'name': 'agg1',
             'description': 'agg2',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 1,
             'name': 'agg2',
             'description': 'agg2',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 2,
             'name': '2.0',
             'description': 'agg1',
             'classifier_values': ['a']}]

        e = self.assemble_disturbance_events_table([event])
        result = sit_disturbance_event_parser.parse(
            e, classifiers, classifier_values, aggregates,
            self.get_mock_disturbance_types(), self.get_mock_age_classes())
        self.assertTrue(list(result.classifier1) == ["1"])
        self.assertTrue(list(result.classifier2) == ["2.0"])

    def test_incorrect_number_of_classifiers_error(self):
        """checks that the format has the correct number of columns
        according to the defined classifiers
        """
        event = {
            "classifier_set": ["a", "?", "EXTRA"],
            "age_eligibility": ["False", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(), self.get_mock_age_classes())

    def test_undefined_classifier_value_error(self):
        """checks that the format has values that are either wildcards or
        classifier sets drawn from the set of defined classifiers values
        and aggregates
        """
        event = {
            "classifier_set": ["UNDEFINED", "?"],
            "age_eligibility": ["False", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_differing_hw_sw_age_criteria_error(self):
        """check that an error is raised if hw age and sw age criteria
        differ (CBM only has stand age)
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["False", -1, 10, -1, 20],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_undefined_age_class_error(self):
        """check that an error is raised if the using_age_class is set
        to true and any of the age class ids are not valid
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, "UNDEFINED"],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_undefined_sort_type_error(self):
        """check if an error is raised when an invalid sort type is specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "UNDEFINED", "A", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_undefined_target_type_error(self):
        """check if an error is raised when an invalid target type is
        specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "UNDEFINED", 100, "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_invalid_target_value_error(self):
        """check if an error is raised when an invalid target value is
        specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", "invalid", "dist1", 2, 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_invalid_disturbance_year_value_error(self):
        """check if an error is raised when an invalid target value is
        specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", "1", "dist1", "invalid", 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_undefined_disturbance_type_value_error(self):
        """check if an error is raised when an undefined disturbance type
        value is specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, "1", "A", "1", "UNDEFINED_DIST", "1", 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_undefined_sort_type_value_error(self):
        """check if an error is raised when an undefined disturbance type
        value is specified
        """
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": ["True", -1, -1, -1, -1],
            "eligibility": [-1] * self.get_num_eligibility_cols(),
            "target": [1.0, 99999999, "A", "1", "dist1", "1", 100]}

        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            e = self.assemble_disturbance_events_table([event])
            sit_disturbance_event_parser.parse(
                e, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_parse_eligibilities(self):
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": [],
            "eligibility": [1],
            "target": [1.0, 1, "A", "1", "dist1", "1", 100]}
        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        e = self.assemble_disturbance_events_table([event])
        sit_events = sit_disturbance_event_parser.parse(
            e, classifiers, classifier_values, aggregates,
            self.get_mock_disturbance_types(),
            None, separate_eligibilities=True)
        elgibilities_input = pd.DataFrame(
            columns=["id", "pool_filter", "state_filter"],
            data=[
                [1, "pool_expression_1", "state_expression_1"]]
        )
        sit_eligibilities = sit_disturbance_event_parser.parse_eligibilities(
            sit_events, elgibilities_input)
        self.assertTrue(
            list(sit_eligibilities.columns) ==
            [x["name"] for x
             in sit_format.get_disturbance_eligibility_format()])

    def test_parse_eligibilities_error_on_missing_id(self):
        event = {
            "classifier_set": ["a", "?"],
            "age_eligibility": [],
            "eligibility": [2],  # missing
            "target": [1.0, 1, "A", "1", "dist1", "1", 100]}
        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        e = self.assemble_disturbance_events_table([event])
        sit_events = sit_disturbance_event_parser.parse(
            e, classifiers, classifier_values, aggregates,
            self.get_mock_disturbance_types(),
            None, separate_eligibilities=True)
        elgibilities_input = pd.DataFrame(
            columns=["id", "pool_filter", "state_filter"],
            data=[
                [1, "", ""]]
        )
        with self.assertRaises(ValueError):
            sit_disturbance_event_parser.parse_eligibilities(
                sit_events, elgibilities_input)
