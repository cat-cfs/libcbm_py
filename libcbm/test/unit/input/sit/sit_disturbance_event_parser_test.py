import unittest
from libcbm.input.sit import sit_disturbance_event_parser


class SITDisturbanceEventParserTest(unittest.TestCase):

    def test_incorrect_number_of_classifiers_error(self):
        """checks that the format has the correct number of columns
        according to the defined classifiers
        """
        self.fail()

    def test_undefined_classifier_value_error(self):
        """checks that the format has values that are either wildcards or
        classifier sets drawn from the set of defined classifiers values
        and aggregates
        """
        self.fail()

    def test_differing_hw_sw_age_criteria_error(self):
        """check that an error is raised if hw age and sw age criteria
        differ (CBM only has stand age)
        """
        self.fail()

    def test_undefined_age_class_error(self):
        """check that an error is raised if the using_age_class is set
        to true and any of the age class ids are not valid
        """
        self.fail()

    def test_undefined_sort_type_error(self):
        """check if an error is raised when an invalid sort type is specified
        """
        self.fail()

    def test_undefined_target_type_error(self):
        """check if an error is raised when an invalid target type is
        specified
        """
        self.fail()

    def test_invalid_target_value_error(self):
        """check if an error is raised when an invalid target value is
        specified
        """
        self.fail()

    def test_invalid_disturbance_year_value_error(self):
        """check if an error is raised when an invalid target value is
        specified
        """
        self.fail()

    def test_undefined_disturbance_type_value_error(self):
        """check if an error is raised when an undefined disturbance type
        value is specified
        """
        self.fail()


