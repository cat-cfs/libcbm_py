import unittest
import os
import json
from libcbm import resources
from libcbm.input.sit import sit_reader


class SITReaderTest(unittest.TestCase):
    def test_error_on_unsupported_config_type(self):
        config = {
            "classifiers": {
                "type": "NOT SUPPORTED",
                "params": {"p1": "classifiers"},
            }
        }
        with self.assertRaises(NotImplementedError):
            sit_reader.read(config, ".")

    def test_read_csv_integration(self):
        data_dir = os.path.join(
            resources.get_test_resources_dir(), "cbm3_tutorial2"
        )
        config_path = os.path.join(data_dir, "sit_config.json")
        with open(config_path) as config_file:
            config = json.load(config_file)["import_config"]
        result = sit_reader.read(config, data_dir)
        expected_tables = [
            "classifiers",
            "classifier_values",
            "classifier_aggregates",
            "disturbance_types",
            "age_classes",
            "inventory",
            "yield_table",
            "disturbance_events",
            "transition_rules",
        ]
        for table in expected_tables:
            self.assertTrue(result.__dict__[table] is not None)

    def test_read_excel_integration(self):
        data_dir = os.path.join(
            resources.get_test_resources_dir(), "cbm3_tutorial2_eligibilities"
        )
        config_path = os.path.join(data_dir, "sit_config.json")
        with open(config_path) as config_file:
            config = json.load(config_file)["import_config"]
        result = sit_reader.read(config, data_dir)
        expected_tables = [
            "classifiers",
            "classifier_values",
            "classifier_aggregates",
            "disturbance_types",
            "age_classes",
            "inventory",
            "yield_table",
            "disturbance_events",
            "transition_rules",
            "eligibilities",
        ]
        for table in expected_tables:
            self.assertTrue(result.__dict__[table] is not None)
