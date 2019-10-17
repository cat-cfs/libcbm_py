import unittest
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import DEFAULT
import pandas as pd
from mock import Mock
import contextlib

from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import SITEventProcessor


class SITEventProcessorTest(unittest.TestCase):

    def test_process_events_behaviour(self):

        patch_path = "libcbm.model.cbm.rule_based.sit.sit_event_processor"
        # patch all of the imported modules so they can be mocked
        with patch.multiple(
                patch_path,
                event_processor=DEFAULT,
                rule_filter=DEFAULT,
                rule_target=DEFAULT,
                sit_stand_filter=DEFAULT,
                sit_stand_target=DEFAULT,
                cbm_variables=DEFAULT) as mocks:

            mock_sit_events = pd.DataFrame(
                data={
                    "time_step": [1, 1, 1, 2, 2, 2],
                    "disturbance_type": [
                        "fire", "fire", "harvest", "harvest", "fire", "fire"]
                })
            mock_classifiers = pd.DataFrame()
            mock_inventory = pd.DataFrame()
            mock_pools = pd.DataFrame()
            mock_state_variables = pd.DataFrame()

            # mock filter
            rule_filter = Mock()
            rule_filter.merge_filters = Mock()
            rule_filter.create_filter = Mock()
            rule_filter.evaluate_filter = Mock()

            # mock rule target
            mocks["rule_target"].compute_disturbance_production = Mock()

            # mock event processor
            mocks["event_processor"].process_event = Mock()

            # mock cbm defaults reference
            mock_cbm_defaults_ref = Mock()
            mock_cbm_defaults_ref.get_disturbance_types = Mock()
            mock_cbm_defaults_ref.get_disturbance_types.side_effect = \
                lambda: [
                    {"disturbance_type_name": "fire", "disturbance_type_id": 1},
                    {"disturbance_type_name": "harvest", "disturbance_type_id": 2}]

            mock_cbm_defaults_ref.get_flux_indicators = Mock()
            mock_cbm_defaults_ref.get_flux_indicators.side_effect = \
                lambda: ["a", "b", "c"]

            # mock classifier filter
            mock_classifier_filter_builder = Mock()
            mock_classifier_filter_builder.create_classifiers_filter = Mock()

            # these mocks are not called by SITEventProcessor, but are
            # passed to underlying functions or objects
            mock_model_functions = "mock_model_functions"
            mock_compute_functions = "mock_compute_functions"
            mock_random_generator = "mock_random_generator"
            mock_on_unrealized_event = "mock_unrealized_event"

            sit_event_processor = SITEventProcessor(
                model_functions=mock_model_functions,
                compute_functions=mock_compute_functions,
                cbm_defaults_ref=mock_cbm_defaults_ref,
                classifier_filter_builder=mock_classifier_filter_builder,
                random_generator=mock_random_generator,
                on_unrealized_event=mock_on_unrealized_event)

            (dist_types, classifiers, inventory, pools, state_variables) = \
                sit_event_processor.process_events(
                    time_step=1,
                    sit_events=mock_sit_events,
                    classifiers=mock_classifiers,
                    inventory=mock_inventory,
                    pools=mock_pools,
                    state_variables=mock_state_variables)
