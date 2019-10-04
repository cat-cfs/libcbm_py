import unittest
import pandas as pd
from libcbm.input.sit import sit_stand_filter


class SITStandFilterTest(unittest.TestCase):

    def test_expected_result_on_create_age_state_variable_filter(self):

        mock_data = [
            [10, 10],
            [-1, 10],
            [10, -1],
            [-1, -1]]

        expected_expressions = [
            "(age >= 10) & (age <= 10)",
            "(age <= 10)",
            "(age >= 10)",
            ""]

        expected_columns = [
            ["min_age", "max_age"],
            ["max_age"],
            ["min_age"],
            []]

        mock_sit_transitions_data = pd.DataFrame(
            data=mock_data,
            columns=["min_age", "max_age"])

        mock_state_variables = pd.DataFrame({"age": list(range(0, 10))})

        def create_test_func(i_row):
            def mock_create_filter(expression, state_variables, columns):
                self.assertTrue(state_variables.equals(mock_state_variables))
                self.assertTrue(expression == expected_expressions[i_row])
                self.assertTrue(columns == expected_columns[i_row])
            return mock_create_filter

        rows = mock_sit_transitions_data.to_dict("records")
        for i_row, row in enumerate(rows):
            sit_stand_filter.create_state_variable_filter(
                create_test_func(i_row), row, mock_state_variables,
                sit_stand_filter.get_state_variable_age_filter_mappings())







    def test_expected_result_on_create_classifier_filter(self):
        pass