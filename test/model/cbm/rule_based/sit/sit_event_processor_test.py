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


        mock_rule_filter_functions = Mock()
        mock_rule_target_functions = Mock()
        mock_event_processor_functions = Mock()
        mock_model_functions = Mock()
        mock_compute_functions = Mock()
        mock_cbm_defaults_ref = Mock()
        mock_classifier_filter_builder = Mock()
        mock_random_generator = Mock()
        mock_on_unrealized_event = Mock()

        sit_event_processor = SITEventProcessor(
            rule_filter_functions=mock_rule_filter_functions,
            rule_target_functions=mock_rule_target_functions,
            event_processor_functions=mock_event_processor_functions,
            model_functions=mock_model_functions,
            compute_functions=mock_compute_functions,
            cbm_defaults_ref=mock_cbm_defaults_ref,
            classifier_filter_builder=mock_classifier_filter_builder,
            random_generator=mock_random_generator,
            on_unrealized_event=mock_on_unrealized_event
        )

        (dist_types, classifiers, inventory, pools, state_variables) = \
            sit_event_processor.process_events(
                time_step=1,
                sit_events=mock_sit_events,
                classifiers=mock_classifiers,
                inventory=mock_inventory,
                pools=mock_pools,
                state_variables=mock_state_variables)