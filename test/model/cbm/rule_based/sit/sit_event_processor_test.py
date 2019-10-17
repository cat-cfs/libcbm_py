import unittest
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import SITEventProcessor


class SITEventProcessorTest(unittest.TestCase):

    def test_process_events_behaviour(self):

        mock_sit_events = pd.DataFrame()
        mock_classifiers = pd.DataFrame()
        mock_inventory = pd.DataFrame()
        mock_pools = pd.DataFrame()
        mock_state_variables = pd.DataFrame()

        # mock filter
        mock_rule_filter_functions = Mock()
        mock_rule_filter_functions.merge_filters = Mock()
        mock_rule_filter_functions.create_filter = Mock()
        mock_rule_filter_functions.evaluate_filter = Mock()

        # mock rule target
        mock_rule_target_functions = Mock()
        mock_rule_target_functions.compute_disturbance_production = Mock()

        # mock event processor
        mock_event_processor_functions = Mock()
        mock_event_processor_functions.process_event = Mock()

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
            rule_filter_functions=mock_rule_filter_functions,
            rule_target_functions=mock_rule_target_functions,
            event_processor_functions=mock_event_processor_functions,
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
