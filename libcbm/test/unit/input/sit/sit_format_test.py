import unittest
from libcbm.input.sit import sit_format


class SITFormatTest(unittest.TestCase):

    def test_get_classifier_format(self):

        with self.assertRaises(ValueError):
            sit_format.get_classifier_format(2)
