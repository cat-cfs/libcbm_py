import unittest
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter


class ClassifierFilterTest(unittest.TestCase):

    def test_create_classifiers_filter_expected_value(self):

        classifier_set = ["c1_v1", "?", "agg1"]

        classifiers_config = {
            "classifiers": [
                {"id": 1, "name": "c1"},
                {"id": 2, "name": "c2"},
                {"id": 3, "name": "c3"}
            ],
            "classifier_values": [
                {"id": 1, "classifier_id": 1, "name": "c1_v1"},
                {"id": 2, "classifier_id": 1, "name": "c1_v2"},
                {"id": 3, "classifier_id": 2, "name": "c2_v1"},
                {"id": 4, "classifier_id": 2, "name": "c2_v2"},
                {"id": 5, "classifier_id": 3, "name": "c3_v1"},
                {"id": 6, "classifier_id": 3, "name": "c3_v2"},
                {"id": 7, "classifier_id": 3, "name": "c3_v3"}
            ]
        }
        classifier_aggregates = [
            {'classifier_id': 3,
             'name': 'agg1',
             'description': 'agg1',
             'classifier_values': ['c3_v1', 'c3_v3']}]

        rule_filter = ClassifierFilter(
            classifiers_config, classifier_aggregates)

        def get_classifier_value_index(classifier_id):
            return {
                x["name"]: x["id"] for x
                in classifiers_config["classifier_values"]
                if x["classifier_id"] == classifier_id}

        c1 = get_classifier_value_index(1)
        c2 = get_classifier_value_index(2)
        c3 = get_classifier_value_index(3)

        classifier_values = pd.DataFrame(
            [(c1[x[0]], c2[x[1]], c3[x[2]])
             for x in [
                ("c1_v1", "c2_v1", "c3_v3"),  # match
                ("c1_v2", "c2_v1", "c3_v3"),  # non-match (c1_v2)
                ("c1_v1", "c2_v2", "c3_v3"),  # match
                ("c1_v1", "c2_v2", "c3_v1"),  # match
                ("c1_v1", "c2_v2", "c3_v2"),  # non-match (aggregate)
             ]], columns=["c1", "c2", "c3"])

        result = rule_filter.create_classifiers_filter(
            classifier_set, classifier_values)
        self.assertTrue(
            result.expression == "c_0 == 1 & (c_2 == 5 | c_2 == 7)")
        self.assertTrue(list(result.local_dict["c_0"]) == [1, 2, 1, 1, 1])
        self.assertTrue(list(result.local_dict["c_1"]) == [3, 3, 4, 4, 4])
        self.assertTrue(list(result.local_dict["c_2"]) == [7, 7, 7, 5, 6])
