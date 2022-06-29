import unittest
from types import SimpleNamespace
import pandas as pd
from libcbm.storage import dataframe
from libcbm.model.cbm.rule_based import rule_filter


class RuleFilterTest(unittest.TestCase):
    def test_evaluate_filter_none_result(self):
        mock_filter = SimpleNamespace()
        mock_filter.expression = None
        mock_filter.data = None
        self.assertTrue(rule_filter.evaluate_filters(mock_filter) is None)
        self.assertTrue(rule_filter.evaluate_filters(None) is None)

    def test_create_filter_expected_output(self):
        data = dataframe.from_pandas(
            pd.DataFrame(
                {"A": range(1, 10), "B": range(1, 10), "C": range(1, 10)}
            )
        )
        result = rule_filter.create_filter("(A < 5) | (B > 6)", data)

        self.assertTrue(result.expression == "(A < 5) | (B > 6)")
        self.assertTrue(result.data.to_pandas().equals(data.to_pandas()))

    def test_evaluate_filter_expected_output(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                "(a < 5) | (b > 6)",
                dataframe.from_pandas(
                    pd.DataFrame(
                        {
                            "a": range(1, 10),
                            "b": range(1, 10),
                            "c": range(1, 10),
                        }
                    )
                ),
            )
        )
        self.assertTrue(
            result.to_list() == [True] * 4 + [False] * 2 + [True] * 3
        )

    def test_evaluate_filter_expected_output2(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                "a == 1",
                dataframe.from_pandas(pd.DataFrame({"a": [0, 1, 0, 1]})),
            ),
            rule_filter.create_filter(
                "a == 1",
                dataframe.from_pandas(pd.DataFrame({"a": [0, 0, 1, 1]})),
            ),
        )

        self.assertTrue(result.to_list() == [False, False, False, True])

    def test_error_on_invalid_expression_evaluate_filter(self):
        with self.assertRaises(pd.core.computation.ops.UndefinedVariableError):
            rule_filter.evaluate_filters(
                rule_filter.create_filter(
                    "(A < 5) | (C > 6)",  # note upper vs lower case below
                    dataframe.from_pandas(
                        pd.DataFrame({"a": range(1, 10), "c": range(1, 10)})
                    ),
                )
            )

    def test_true_series_retured_with_null_or_empty_strexpressions(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                None,
                dataframe.from_pandas(
                    pd.DataFrame({"a": range(1, 10), "c": range(1, 10)})
                ),
            ),
            rule_filter.create_filter(
                "",
                dataframe.from_pandas(
                    pd.DataFrame({"f": range(1, 10), "g": range(1, 10)})
                ),
            ),
        )
        self.assertTrue(result.to_list() == [True] * 9)

    def test_error_raised_with_non_matching_row_dimension(self):
        with self.assertRaises(ValueError):
            rule_filter.evaluate_filters(
                rule_filter.create_filter(
                    "a == 2",
                    dataframe.from_pandas(
                        pd.DataFrame({"a": range(1, 5), "c": range(1, 5)})
                    ),
                ),
                rule_filter.create_filter(
                    "g < 2",
                    dataframe.from_pandas(
                        pd.DataFrame({"f": range(1, 10), "g": range(1, 10)})
                    ),
                ),
            )

    def test_none_retured_with_no_data_specified(self):
        result = rule_filter.evaluate_filters(
            rule_filter.create_filter(
                "a",
                None,
            ),
            rule_filter.create_filter(
                "",
                None,
            ),
        )
        self.assertTrue(result is None)
