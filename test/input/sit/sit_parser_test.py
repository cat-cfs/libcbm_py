import unittest
import pandas as pd
from libcbm.input.sit import sit_parser


class SITParserTest(unittest.TestCase):
    def test_unpack_table_raises_on_duplicate_column(self):
        """checks that if any 2 columns are the identical, an error is thrown"""
        with self.assertRaises(ValueError):
            sit_parser.unpack_table(
                table=pd.DataFrame([("0", "0")]),
                column_descriptions=[
                    {"index": 0, "name": "duplicate"},
                    {"index": 1, "name": "duplicate"},
                ],
                table_name="",
            )

    def test_unpack_table_expected_result(self):
        """test that unpack_table function returns an expected value"""
        unpacked = sit_parser.unpack_table(
            table=pd.DataFrame([("1", "2", "3")]),
            column_descriptions=[
                {"index": 0, "name": "col0", "type": int},
                {"index": 1, "name": "col1", "type": float},
                {"index": 2, "name": "col2"},
            ],
            table_name="",
        )
        self.assertTrue(list(unpacked.columns) == ["col0", "col1", "col2"])
        table = list(unpacked.itertuples())[0]

        self.assertTrue(table.col0 == 1)
        self.assertTrue(table.col1 == 2.0)
        self.assertTrue(table.col2 == "3")

    def test_unpack_column_raises_on_unconvertable_value(self):
        """checks that unpacking a column with a type constraint works"""

        cases = [
            ("invalid_integer", int),
            ("1.1", int),
            ("invalid_float", float),
        ]

        for value, constraint_type in cases:
            with self.assertRaises(ValueError):
                sit_parser.unpack_table(
                    table=pd.DataFrame([(value,)]),
                    column_descriptions=[
                        {"index": 0, "name": "col0", "type": constraint_type}
                    ],
                    table_name="",
                )

    def test_unpack_column_raises_on_min_value_violation(self):
        """checks that unpacking a column with a min value constraint works"""
        with self.assertRaises(ValueError):
            sit_parser.unpack_table(
                table=pd.DataFrame([(-1,)]),
                column_descriptions=[
                    {"index": 0, "name": "col0", "type": int, "min_value": 0}
                ],
                table_name="",
            )

    def test_unpack_column_raises_on_max_value_violation(self):
        """checks that unpacking a column with a max value constraint works"""
        with self.assertRaises(ValueError):
            sit_parser.unpack_table(
                table=pd.DataFrame([(1,)]),
                column_descriptions=[
                    {"index": 0, "name": "col0", "type": int, "max_value": 0}
                ],
                table_name="",
            )

    def test_parse_bool_func(self):
        """test the parse bool function/function generator"""
        parse_bool = sit_parser.get_parse_bool_func("", "")

        for x in ["t", "T", "Y", "y", "tRuE", 1, 999, True]:
            self.assertTrue(parse_bool(x))

        for x in ["f", "F", "N", "n", "FaLsE", -1, -999, 0, False]:
            self.assertFalse(parse_bool(x))

        for x in ["invalid", 10.1, -100.5]:
            with self.assertRaises(ValueError):
                parse_bool(x)
