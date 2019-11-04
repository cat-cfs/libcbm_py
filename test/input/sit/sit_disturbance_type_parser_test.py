import unittest
import pandas as pd
from libcbm.input.sit import sit_disturbance_type_parser


class SITDisturbanceTypeParserTest(unittest.TestCase):

    def test_check_expected_result(self):
        """check a valid parsed result"""
        result = sit_disturbance_type_parser.parse(
            pd.DataFrame(data=[
                ("1", "fire"),
                (2, "clearcut"),
                ("3", "clearcut")
            ]))

        self.assertTrue(list(result.id) == ["1", "2", "3"])
        self.assertTrue(list(result.name) == ["fire", "clearcut", "clearcut"])

    def test_duplicate_id_error(self):
        """check if an error is raise on duplicate ids
        """
        with self.assertRaises(ValueError):
            sit_disturbance_type_parser.parse(
                pd.DataFrame(data=[
                    ("1", "fire"),
                    (1, "clearcut"),
                    ("3", "clearcut")
                ]))
