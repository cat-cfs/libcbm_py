import unittest
import pandas as pd
from libcbm.model.cbm.rule_based.sit import sit_stand_filter


class SITStandFilterTest(unittest.TestCase):
    def test_expected_result_on_create_age_state_variable_filter(self):
        mock_data = [[10, 10], [-1, 10], [10, -1], [-1, -1]]

        expected_expressions = [
            "(age >= 10) & (age <= 10)",
            "(age <= 10)",
            "(age >= 10)",
            "",
        ]

        mock_sit_transitions_data = pd.DataFrame(
            data=mock_data, columns=["min_age", "max_age"]
        )

        rows = mock_sit_transitions_data.to_dict("records")
        for i_row, row in enumerate(rows):
            exp = sit_stand_filter.create_state_filter_expression(
                row, sit_stand_filter.get_state_variable_age_filter_mappings()
            )
            self.assertTrue(exp == expected_expressions[i_row])

    def test_create_last_disturbance_type_filter(self):
        exp = sit_stand_filter.create_last_disturbance_type_filter(
            sit_data={"LastDistTypeID": "distid1"},
            sit_disturbance_type_map={"distid1": 9999},
        )
        self.assertTrue(exp == "(last_disturbance_type == 9999)")

    def test_expected_result_on_create_state_variable_filter(self):
        mock_data_columns = [
            "min_age",
            "max_age",
            "MinYearsSinceDist",
            "MaxYearsSinceDist",
        ]

        mock_data = [
            [10, -1, -1, 5],
            [-1, 7, -1, -1],
            [10, -1, 30, -1],
            [-1, -1, -1, -1],
        ]

        expected_expressions = [
            "(age >= 10) & (time_since_last_disturbance <= 5)",
            "(age <= 7)",
            "(age >= 10) & (time_since_last_disturbance >= 30)",
            "",
        ]

        mock_sit_transitions_data = pd.DataFrame(
            data=mock_data, columns=mock_data_columns
        )

        rows = mock_sit_transitions_data.to_dict("records")
        for i_row, row in enumerate(rows):
            exp = sit_stand_filter.create_state_filter_expression(row, False)
            self.assertTrue(exp == expected_expressions[i_row])

    def test_get_pool_variable_filter_mappings_expected_value(self):
        pool_mappings = sit_stand_filter.get_pool_variable_filter_mappings()
        mock_data_columns = [x[0] for x in pool_mappings]

        mock_data = [
            list(range(0, len(pool_mappings))),
            [-1 for _ in pool_mappings],
        ]

        mock_sit_events = pd.DataFrame(
            data=mock_data, columns=mock_data_columns
        )
        expected_expression_tokens = [
            "({pool_exp} {operator} {value})".format(
                pool_exp="({})".format(" + ".join(x[1])),
                operator=x[2],
                value=i_x,
            )
            for i_x, x in enumerate(pool_mappings)
        ]

        expected_expression = " & ".join(expected_expression_tokens)

        expected_columns = set()
        for x in pool_mappings:
            expected_columns.update(set(x[1]))

        rows = mock_sit_events.to_dict("records")
        expression0 = sit_stand_filter.create_pool_filter_expression(rows[0])
        self.assertTrue(expression0 == expected_expression)

        expression1 = sit_stand_filter.create_pool_filter_expression(rows[1])
        self.assertTrue(expression1 == "")

    def test_get_classifier_set_expected_result(self):
        mock_classifiers = ["a", "b", "c"]
        mock_classifier_sets = [
            ["a1", "b1", "c1"],
            ["a2", "b2", "c2"],
            ["a3", "b3", "c3"],
        ]
        mock_sit_data = pd.DataFrame(
            data=mock_classifier_sets, columns=mock_classifiers
        )

        for i_row, row in mock_sit_data.iterrows():
            result = sit_stand_filter.get_classifier_set(row, mock_classifiers)
            self.assertTrue(result == mock_classifier_sets[i_row])
