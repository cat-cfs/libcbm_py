import unittest
from libcbm.input.sit import sit_parser

class SITParserTest(unittest.TestCase):

    def test_unpack_table_raises_on_duplicate_column(self):
        """checks that if any 2 columns are the identical, an error is thrown
        """
        self.fail()

    def test_unpack_table_expected_result(self):
        """test that unpack_table function returns an expected value
        """
        self.fail()

    def test_unpack_column_raises_on_unconvertable_value(self):
        """checks that unpacking a column with a type constraint works
        """
        self.fail()

    def test_unpack_column_raises_on_min_value_violation(self):
        """checks that unpacking a column with a min value constraint works
        """
        self.fail()

    def test_unpack_column_raises_on_max_value_violation(self):
        """checks that unpacking a column with a max value constraint works
        """
        self.fail()

    def test_parse_bool_func(self):
        """test the parse bool function/function generator
        """
        self.fail()