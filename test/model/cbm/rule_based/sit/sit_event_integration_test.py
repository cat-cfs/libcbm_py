import unittest
import os
import json
import pandas as pd
import numpy as np
from types import SimpleNamespace
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit import sit_reader
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_age_class_parser
from libcbm.input.sit import sit_disturbance_type_parser
from libcbm.model.cbm import cbm_variables

from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit import sit_event_processor

from libcbm import resources


def get_test_data_dir():
    return os.path.join(
        resources.get_examples_dir(), "sit", "rule_based_events")


def load_sit_data(events_file):
    sit = SimpleNamespace()
    sit_config = load_config(events_file)
    sit.sit_data = sit_reader.read(
        sit_config["import_config"], get_test_data_dir())
    sit.config = sit_config
    return sit


def load_config(events_file):
    sit_rule_based_examples_dir = get_test_data_dir()

    config_path = os.path.join(sit_rule_based_examples_dir, "sit_config.json")
    with open(config_path) as sit_config_fp:
        sit_config = json.load(sit_config_fp)

    sit_config["import_config"]["events"] = {
        "type": "csv", "params": {"path": f"{events_file}.csv"}}
    return sit_config


class SITEventIntegrationTest(unittest.TestCase):

    def test_sit_rule_based_event_integration(self):

        sit = load_sit_data("area_target_age_sort")

        sit = sit_cbm_factory.initialize_sit_objects(sit)
        classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
        sit_events = sit_cbm_factory.initialize_events(sit)
        cbm = sit_cbm_factory.initialize_cbm(sit)

        classifier_filter = ClassifierFilter(
            classifiers_config=sit_cbm_factory.get_classifiers(
                sit.sit_data.classifiers, sit.sit_data.classifier_values),
            classifier_aggregates=sit.sit_data.classifier_aggregates)
        processor = sit_event_processor.SITEventProcessor(
            model_functions=cbm.model_functions,
            compute_functions=cbm.compute_functions,
            cbm_defaults_ref=sit.defaults,
            classifier_filter_builder=classifier_filter,
            random_generator=np.random.rand,
            on_unrealized_event=lambda shortfall, sit_event:
                print(
                    f"unrealized target. Shortfall: {shortfall}, "
                    f"Event: {sit_event}"))

        pre_dynamics_func = sit_event_processor.get_pre_dynamics_func(
            processor, sit_events)
        cbm_vars = cbm_variables.initialize_simulation_variables(
            classifiers, inventory, sit.defaults.get_pools(),
            sit.defaults.get_flux_indicators())
        cbm_vars = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

