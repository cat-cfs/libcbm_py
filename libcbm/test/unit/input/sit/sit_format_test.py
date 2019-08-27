import unittest
from libcbm.input.sit import sit_format


class SITFormatTest(unittest.TestCase):

    def test_get_classifier_format_value_error(self):
        with self.assertRaises(ValueError):
            sit_format.get_classifier_format(2)

    def test_get_disturbance_type_format_value_error(self):
        with self.assertRaises(ValueError):
            sit_format.get_disturbance_type_format(1)

        with self.assertRaises(ValueError):
            sit_format.get_disturbance_type_format(4)

    def test_get_yield_format_value_error(self):
        mock_classifier_names = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            sit_format.get_yield_format(mock_classifier_names, 1)

    def test_get_transition_rules_format_value_error(self):
        mock_classifier_names = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            sit_format.get_transition_rules_format(mock_classifier_names, 1)
        with self.assertRaises(ValueError):
            sit_format.get_transition_rules_format(mock_classifier_names, 17)

    def test_basic_formats(self):
        """Checks that all returned formats have sequential indexes and the
        basic name/index properties in all dictionaries
        """
        mock_classifier_names = ["a", "b", "c"]
        n_classifiers = len(mock_classifier_names)
        cases = [
            sit_format.get_classifier_format(3),
            sit_format.get_disturbance_type_format(2),
            sit_format.get_age_class_format(),
            sit_format.get_yield_format(mock_classifier_names, 5),
            sit_format.get_transition_rules_format(
                mock_classifier_names, 2 * n_classifiers +
                len(sit_format.get_age_eligibility_columns(n_classifiers)) +
                4),
            sit_format.get_inventory_format(
                mock_classifier_names, n_classifiers + 6),
            sit_format.get_disturbance_event_format(
                mock_classifier_names, n_classifiers +
                len(sit_format.get_disturbance_eligibility_columns(
                    n_classifiers)) +
                len(sit_format.get_age_eligibility_columns(n_classifiers)) + 7)
            ]
        required_keys = {"name", "index"}
        all_keys = required_keys.union({"min_value", "max_value", "type"})
        for case in cases:
            for i, col in enumerate(case):
                # check that the required keys exist in the column description
                self.assertTrue(required_keys.issubset(col.keys()))
                # check that the only keys in the "all keys" set are in the
                # column description.
                self.assertEqual(len(set(col.keys()).difference(all_keys)), 0)
                self.assertEqual(col["index"], i)
