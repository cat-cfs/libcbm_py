from types import SimpleNamespace
import unittest
import os
import pandas as pd
from unittest.mock import Mock
from unittest.mock import patch

from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm import resources


class SITCBMFactoryTest(unittest.TestCase):

    def test_integration_with_tutorial2(self):
        """tests full CBM integration with rule based disturbances
        """
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2", "sit_config.json")
        sit = sit_cbm_factory.load_sit(config_path)
        classifiers, inventory = \
            sit_cbm_factory.initialize_inventory(sit)
        with sit_cbm_factory.initialize_cbm(sit) as cbm:
            results, reporting_func = \
                cbm_simulator.create_in_memory_reporting_func()
            rule_based_processor = \
                sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)

            cbm_simulator.simulate(
                cbm,
                n_steps=1,
                classifiers=classifiers,
                inventory=inventory,
                pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                reporting_func=reporting_func)
            self.assertTrue(
                results.pools[results.pools.timestep == 0].shape[0] ==
                inventory.shape[0])
            self.assertTrue(
                len(rule_based_processor.sit_event_stats_by_timestep) > 0)

    def test_integration_with_tutorial2_eligbilities(self):
        """tests full CBM integration with rule based disturbances and
        disturbance event eligibility expressions
        """
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2_eligibilities", "sit_config.json")
        sit = sit_cbm_factory.load_sit(config_path)
        classifiers, inventory = \
            sit_cbm_factory.initialize_inventory(sit)
        with sit_cbm_factory.initialize_cbm(sit) as cbm:
            results, reporting_func = \
                cbm_simulator.create_in_memory_reporting_func()
            rule_based_processor = \
                sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)

            cbm_simulator.simulate(
                cbm,
                n_steps=1,
                classifiers=classifiers,
                inventory=inventory,
                pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                reporting_func=reporting_func)
            self.assertTrue(
                results.pools[results.pools.timestep == 0].shape[0] ==
                inventory.shape[0])
            self.assertTrue(
                len(rule_based_processor.sit_event_stats_by_timestep) > 0)

    @patch("libcbm.input.sit.sit_cbm_factory.resources")
    @patch("libcbm.input.sit.sit_cbm_factory.SITCBMDefaults")
    @patch("libcbm.input.sit.sit_cbm_factory.SITMapping")
    def test_classifier_maps(self, SITMapping, SITCBMDefaults, resources):

        classifiers = pd.DataFrame(
            columns=["id", "name"],
            data=[
                [1, "c1"],
                [2, "c2"],
                [3, "c3"]])
        classifier_values = pd.DataFrame(
            columns=["classifier_id", "name"],
            data=[
                [1, "c1v1"],
                [1, "c1v2"],
                [2, "c2v1"],
                [2, "c2v2"],
                [2, "c2v3"],
                [3, "c3v1"],
                [3, "c3v2"],
                [3, "c3v3"],
                [3, "c3v4"]])
        sit = SimpleNamespace(
            config={"mapping_config": None},
            sit_data=SimpleNamespace(
                classifiers=classifiers,
                classifier_values=classifier_values,
                disturbance_types=Mock()
            ),
            sit_mapping=Mock()
        )
        sit_cbm_factory.initialize_sit_objects(sit)
        self.assertTrue(len(sit.classifier_names) == len(classifiers.index))
        self.assertTrue(len(sit.classifier_ids) == len(classifiers.index))
        self.assertTrue(
            len(sit.classifier_value_ids) == len(classifiers.index))
        self.assertTrue(
            len(sit.classifier_value_names) == len(classifier_values.index))
        self.assertTrue(
            set(classifiers.name) == set(sit.classifier_value_ids.keys()))
        self.assertTrue(
            set(classifiers.name) == set(sit.classifier_names.values()))

        for _, classifier_row in classifiers.iterrows():
            classifier_name = classifier_row["name"]
            classifier_id = classifier_row["id"]
            expected_classifier_names = classifier_values[
                classifier_values.classifier_id == classifier_id].name

            self.assertTrue(
                set(sit.classifier_value_ids[classifier_name].keys()) ==
                set(expected_classifier_names))

        self.assertTrue(
            set(classifier_values.name) ==
            set(sit.classifier_value_names.values()))
