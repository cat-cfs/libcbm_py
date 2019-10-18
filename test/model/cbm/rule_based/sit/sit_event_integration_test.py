import unittest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit import sit_reader
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_age_class_parser
from libcbm.model.cbm import cbm_variables

from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit.sit_event_processor import \
    get_pre_dynamics_func
from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import SITEventProcessor


def assemble_disturbance_events_table(events):
    return pd.DataFrame([
        event["classifier_set"] +
        event["age_eligibility"] +
        event["eligibility"] +
        event["target"]
        for event in events
    ])


def get_num_eligibility_cols():
    num_eligibility_cols = len(
        sit_format.get_disturbance_eligibility_columns(0))
    return num_eligibility_cols


class SITEventIntegrationTest(unittest.TestCase):

    def test_sit_rule_based_event_integration(self):

        sit = SimpleNamespace()
        sit.config = {
            "mapping_config": {
                "nonforest": None,
                "species": {
                    "species_classifier": "classifier1",
                    "species_mapping": [
                        {
                            "user_species": "a",
                            "default_species": "Spruce"
                        }
                    ]
                },
                "spatial_units": {
                    "mapping_mode": "SingleDefaultSpatialUnit",
                    "default_spuid": 42
                },
                "disturbance_types": [
                    {
                        "user_dist_type": "fire",
                        "default_dist_type": "Wildfire"
                    }
                ]
            }
        }
        events = [{
            "classifier_set": ["a"],
            "age_eligibility": ["False", -1, -1, -1, -1],
            "eligibility": [-1] * get_num_eligibility_cols(),
            "target": [1.0, "1", "A", 100, "fire", 2, 100]}]

        sit.sit_data = sit_reader.parse(
            sit_classifiers=pd.DataFrame(
                data=[
                    (1, "_CLASSIFIER", "classifier1"),
                    (1, "a", "a")]),
            sit_disturbance_types=pd.DataFrame(data=[
                ("fire", "fire")]),
            sit_age_classes=sit_age_class_parser.generate_sit_age_classes(
                5, 100),
            sit_inventory=pd.DataFrame(
                data=[("a", False, 100, 1, 0, 0, "fire", "fire")]),
            sit_yield=pd.DataFrame([
                ["a", "a"] +
                [x*15 for x in range(0, 20+1)]]),
            sit_events=assemble_disturbance_events_table(events),
            sit_transitions=None
        )
        sit = sit_cbm_factory.initialize_sit_objects(sit)
        classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
        cbm = sit_cbm_factory.initialize_cbm(sit)

        classifier_filter = ClassifierFilter(
            classifiers_config=sit_cbm_factory.get_classifiers(
                sit.sit_data.classifiers, sit.sit_data.classifier_values),
            classifier_aggregates=sit.sit_data.classifier_aggregates)
        sit_event_processor = SITEventProcessor(
            model_functions=cbm.model_functions,
            compute_functions=cbm.compute_functions,
            cbm_defaults_ref=sit.defaults,
            classifier_filter_builder=classifier_filter,
            random_generator=np.random.rand,
            on_unrealized_event=lambda x: None)

        pre_dynamics_func = get_pre_dynamics_func(
            sit_event_processor, sit.sit_data.disturbance_events)
        cbm_vars = cbm_variables.initialize_simulation_variables(
            classifiers, inventory, sit.defaults.get_pools(),
            sit.defaults.get_flux_indicators())
        cbm_vars = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)
