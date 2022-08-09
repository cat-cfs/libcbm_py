from types import SimpleNamespace
import unittest
import os
import pandas as pd
from unittest.mock import Mock
from unittest.mock import patch

from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit.sit_cbm_factory import EventSort
from libcbm.model.cbm import cbm_simulator
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm import resources


class SITCBMFactoryTest(unittest.TestCase):
    def test_integration_with_tutorial2(self):
        """tests full CBM integration with rule based disturbances"""
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2",
            "sit_config.json",
        )
        sit = sit_cbm_factory.load_sit(config_path)
        classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
        with sit_cbm_factory.initialize_cbm(sit) as cbm:
            in_memory_output = CBMOutput()

            rule_based_processor = (
                sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
            )

            cbm_simulator.simulate(
                cbm,
                n_steps=1,
                classifiers=classifiers,
                inventory=inventory,
                pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                reporting_func=in_memory_output.append_simulation_result,
            )
            self.assertTrue(
                in_memory_output.pools.filter(
                    in_memory_output.pools["timestep"] == 0
                ).n_rows
                == inventory.n_rows
            )
            self.assertTrue(
                len(rule_based_processor.sit_event_stats_by_timestep) > 0
            )

    def test_integration_with_tutorial2_eligbilities(self):
        """tests full CBM integration with rule based disturbances and
        disturbance event eligibility expressions
        """
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2_eligibilities",
            "sit_config.json",
        )
        sit = sit_cbm_factory.load_sit(config_path)
        classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
        with sit_cbm_factory.initialize_cbm(sit) as cbm:
            in_memory_cbm_output = CBMOutput()
            rule_based_processor = (
                sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
            )

            cbm_simulator.simulate(
                cbm,
                n_steps=1,
                classifiers=classifiers,
                inventory=inventory,
                pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                reporting_func=in_memory_cbm_output.append_simulation_result,
            )
            self.assertTrue(
                in_memory_cbm_output.pools.filter(
                    in_memory_cbm_output.pools["timestep"] == 0
                ).n_rows
                == inventory.n_rows
            )
            self.assertTrue(
                len(rule_based_processor.sit_event_stats_by_timestep) > 0
            )

    @patch("libcbm.input.sit.sit_cbm_factory.resources")
    @patch("libcbm.input.sit.sit_cbm_factory.SITCBMDefaults")
    @patch("libcbm.input.sit.sit_cbm_factory.SITMapping")
    def test_sit_maps(self, SITMapping, SITCBMDefaults, resources):

        sit_mapping = Mock()
        sit_mapping.get_default_disturbance_type_id.side_effect = (
            lambda x: x.map({"a": "default_a", "b": "default_b"})
        )
        SITMapping.side_effect = lambda *args, **kwargs: sit_mapping

        classifiers = pd.DataFrame(
            columns=["id", "name"], data=[[1, "c1"], [2, "c2"], [3, "c3"]]
        )
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
                [3, "c3v4"],
            ],
        )
        sit_input = SimpleNamespace(
            config={"mapping_config": None},
            sit_data=SimpleNamespace(
                classifiers=classifiers,
                classifier_values=classifier_values,
                disturbance_types=pd.DataFrame(
                    columns=["sit_disturbance_type_id", "id", "name"],
                    data=[[1, "DISTID1", "a"], [2, "DISTID2", "b"]],
                ),
            ),
        )

        sit = sit_cbm_factory.initialize_sit(
            sit_input.sit_data, sit_input.config
        )
        self.assertTrue(len(sit.classifier_names) == len(classifiers.index))
        self.assertTrue(len(sit.classifier_ids) == len(classifiers.index))
        self.assertTrue(
            len(sit.classifier_value_ids) == len(classifiers.index)
        )
        self.assertTrue(
            len(sit.classifier_value_names) == len(classifier_values.index)
        )
        self.assertTrue(
            set(classifiers.name) == set(sit.classifier_value_ids.keys())
        )
        self.assertTrue(
            set(classifiers.name) == set(sit.classifier_names.values())
        )

        for _, classifier_row in classifiers.iterrows():
            classifier_name = classifier_row["name"]
            classifier_id = classifier_row["id"]
            expected_classifier_names = classifier_values[
                classifier_values.classifier_id == classifier_id
            ].name

            self.assertTrue(
                set(sit.classifier_value_ids[classifier_name].keys())
                == set(expected_classifier_names)
            )

        self.assertTrue(
            set(classifier_values.name)
            == set(sit.classifier_value_names.values())
        )

        self.assertTrue(
            sit.default_disturbance_id_map
            == {0: 0, 1: "default_a", 2: "default_b"}
        )

        self.assertTrue(
            sit.disturbance_id_map == {0: 0, 1: "DISTID1", 2: "DISTID2"}
        )

        self.assertTrue(sit.disturbance_name_map == {0: "", 1: "a", 2: "b"})

    def test_integration_with_different_event_sort_modes(self):
        """tests full CBM integration with rule based disturbances and
        disturbance event eligibility expressions
        """
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2_eligibilities",
            "sit_config.json",
        )
        sit = sit_cbm_factory.load_sit(config_path)
        with sit_cbm_factory.initialize_cbm(sit) as cbm:

            rule_based_processor1 = (
                sit_cbm_factory.create_sit_rule_based_processor(
                    sit, cbm, event_sort=EventSort.disturbance_type
                )
            )

            self.assertTrue(
                list(rule_based_processor1.sit_events["sort_field"])
                == list(
                    sit.sit_mapping.get_sit_disturbance_type_id(
                        sit.sit_data.disturbance_events["disturbance_type"]
                    )
                )
            )

            rule_based_processor2 = (
                sit_cbm_factory.create_sit_rule_based_processor(
                    sit, cbm, event_sort=EventSort.default_disturbance_type_id
                )
            )

            expected_default_types = list(
                sit.sit_mapping.get_default_disturbance_type_id(
                    sit.sit_data.disturbance_events[
                        ["disturbance_type"]
                    ].merge(
                        sit.sit_data.disturbance_types,
                        how="left",
                        left_on="disturbance_type",
                        right_on="id",
                    )[
                        "name"
                    ]
                )
            )
            assert (
                list(rule_based_processor2.sit_events["sort_field"])
                == expected_default_types
            )

            rule_based_processor3 = (
                sit_cbm_factory.create_sit_rule_based_processor(
                    sit, cbm, event_sort=EventSort.natural_order
                )
            )

            self.assertTrue(
                list(rule_based_processor3.sit_events["sort_field"])
                == list(range(len(sit.sit_data.disturbance_events.index)))
            )
