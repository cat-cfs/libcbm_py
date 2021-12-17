import unittest
import pandas as pd
from libcbm.input.sit import sit_transition_rule_parser


class SITTransitionRuleParserTest(unittest.TestCase):

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

    def assemble_transition_table(self, transitions):
        return pd.DataFrame([
            tr["classifier_set_src"] +
            tr["age_eligibility"] +
            tr["disturbance_type"] +
            tr["classifier_set_dest"] +
            tr["post_transition"]
            for tr in transitions
        ])

    def test_expected_result_with_numeric_classifiers(self):
        """Checks that numeric classifiers that appear in transition rules
        data are parsed as strings
        """
        transition = {
            "classifier_set_src": [1, "2.0"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": [2, "?"],
            "post_transition": [0, -1, 100]}
        transition_table = self.assemble_transition_table([transition])
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
                (1, "2", "2"),
                (2, "a", "a")
            ],
            columns=["classifier_id", "name", "description"]
        )
        aggregates = [
            {'classifier_id': 1,
             'name': 'agg1',
             'description': 'agg1',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 1,
             'name': 'agg2',
             'description': 'agg2',
             'classifier_values': ['a', 'b']},
            {'classifier_id': 2,
             'name': '2.0',
             'description': '2.0',
             'classifier_values': ['a']}]
        result = sit_transition_rule_parser.parse(
            transition_table, classifiers, classifier_values, aggregates,
            self.get_mock_disturbance_types(), self.get_mock_age_classes())
        self.assertTrue(result.classifier1[0] == "1")
        self.assertTrue(result.classifier2[0] == "2.0")
        self.assertTrue(result.classifier1_tr[0] == "2")
        self.assertTrue(result.classifier2_tr[0] == "?")

    def test_incorrect_number_of_classifiers_error(self):
        """checks that the format has the correct number of columns
        according to the defined classifiers
        """
        transition = {
            # 3 classifiers is more than the 2 defined in the mock classifier
            # definition
            "classifier_set_src": ["a", "agg1", "?"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?", "?"],
            "post_transition": [0, -1, 100]}
        transition_table = self.assemble_transition_table([transition])
        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            sit_transition_rule_parser.parse(
                transition_table, classifiers, classifier_values, aggregates,
                self.get_mock_disturbance_types(), self.get_mock_age_classes())

    def test_undefined_classifier_value_error(self):
        """checks that the format has values that are either wildcards or
        classifier sets drawn from the set of defined classifiers values
        and aggregates
        """
        transition = {
            "classifier_set_src": ["a", "?"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 100]}

        cases = [
            transition.copy(), transition.copy(), transition.copy(),
            transition.copy()]
        cases[0]["classifier_set_src"][0] = "undefined"
        cases[1]["classifier_set_src"][1] = "undefined"
        # check that undefined values result in error in the transition
        # classifier set
        cases[2]["classifier_set_dest"][0] = "undefined"
        # check that aggregate values result in error in the transition
        # classifier set
        cases[3]["classifier_set_dest"][0] = "agg1"

        for case in cases:
            transition_table = self.assemble_transition_table([case])
            classifiers, classifier_values, aggregates = \
                self.get_mock_classifiers()
            with self.assertRaises(ValueError):
                sit_transition_rule_parser.parse(
                    transition_table, classifiers, classifier_values,
                    aggregates, self.get_mock_disturbance_types(),
                    self.get_mock_age_classes())

    def test_differing_hw_sw_age_criteria_error(self):
        """check that an error is raised if hw age and sw age criteria
        differ (CBM only has stand age)
        """
        transition = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 100]}

        cases = [
            transition.copy(), transition.copy(), transition.copy(),
            transition.copy()]

        cases[0]["age_eligibility"] = [
            "true", "undefined", "age2", "age1", "age2"]
        cases[1]["age_eligibility"] = [
            "true", "age1", "undefined", "age1", "age2"]
        cases[2]["age_eligibility"] = [
            "true", "age1", "age2", "undefined", "age2"]
        cases[3]["age_eligibility"] = [
            "true", "age1", "age2", "age1", "undefined"]

        for case in cases:
            transition_table = self.assemble_transition_table([case])
            classifiers, classifier_values, aggregates = \
                self.get_mock_classifiers()
            with self.assertRaises(ValueError):
                sit_transition_rule_parser.parse(
                    transition_table, classifiers, classifier_values,
                    aggregates, self.get_mock_disturbance_types(),
                    self.get_mock_age_classes())

    def test_undefined_age_class_error(self):
        """check that an error is raised if undefined age class ids are used
        """
        transition = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 100]}

        cases = [transition.copy(), transition.copy()]

        # hw and sw differ with "using age class" feature
        cases[0]["age_eligibility"] = ["true", "age1", "age2", "age0", "age1"]
        # hw and sw differ
        cases[1]["age_eligibility"] = ["false", "2", "10", "1", "10"]

        for case in cases:
            transition_table = self.assemble_transition_table([case])
            classifiers, classifier_values, aggregates = \
                self.get_mock_classifiers()
            with self.assertRaises(ValueError):
                sit_transition_rule_parser.parse(
                    transition_table, classifiers, classifier_values,
                    aggregates, self.get_mock_disturbance_types(),
                    self.get_mock_age_classes())

    def test_undefined_disturbance_type_value_error(self):
        """check if an error is raised when an undefined disturbance type
        value is specified
        """
        transition = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["UNDEFINED"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 100]}

        transition_table = self.assemble_transition_table([transition])
        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            sit_transition_rule_parser.parse(
                transition_table, classifiers, classifier_values,
                aggregates, self.get_mock_disturbance_types(),
                self.get_mock_age_classes())

    def test_percent_value_error(self):
        """check if an error is raised when an invalid percent is specified
        """
        transition = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 100]}

        cases = [transition.copy(), transition.copy()]

        # percent > 100
        cases[0]["post_transition"] = [0, -1, 1000]
        # percent < 0
        cases[1]["post_transition"] = [0, -1, -1]
        # non-numeric percent
        cases[1]["post_transition"] = [0, -1, "percent"]

        for case in cases:
            transition_table = self.assemble_transition_table([case])
            classifiers, classifier_values, aggregates = \
                self.get_mock_classifiers()
            with self.assertRaises(ValueError):
                sit_transition_rule_parser.parse(
                    transition_table, classifiers, classifier_values,
                    aggregates, self.get_mock_disturbance_types(),
                    self.get_mock_age_classes())

    def test_grouped_percent_value_error(self):
        """check if an error is raised when a grouped set of transition rule
        rows percent field does not add to 100%
        """

        # the following pair of transition rules will be grouped since they
        # have:
        #   1. the same src classifier set
        #   2. the same age eligibility,
        #   3. same disturbance type
        # since the summed percent is > 100 an error is expected

        transition1 = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 51]}
        transition2 = {
            "classifier_set_src": ["a", "agg1"],
            "age_eligibility": ["f", "-1", "-1", "-1", "-1"],
            "disturbance_type": ["dist1"],
            "classifier_set_dest": ["b", "?"],
            "post_transition": [0, -1, 50]}

        transition_table = self.assemble_transition_table([
            transition1, transition2])
        classifiers, classifier_values, aggregates = \
            self.get_mock_classifiers()
        with self.assertRaises(ValueError):
            sit_transition_rule_parser.parse(
                transition_table, classifiers, classifier_values,
                aggregates, self.get_mock_disturbance_types(),
                self.get_mock_age_classes())
