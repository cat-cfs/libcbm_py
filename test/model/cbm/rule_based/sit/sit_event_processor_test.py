import unittest
from unittest.mock import patch
from unittest.mock import DEFAULT
from types import SimpleNamespace
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import SITEventProcessor

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.sit.sit_event_processor"


class SITEventProcessorTest(unittest.TestCase):

    def test_process_events_behaviour(self):
        """Test some of the internal behaviour of SITEventProcessor, and check
        the expected result with a pair of events on a single timestep.
        """

        # patch all of the imported modules so they can be mocked
        with patch.multiple(
                PATCH_PATH,
                event_processor=DEFAULT,
                rule_filter=DEFAULT,
                rule_target=DEFAULT,
                sit_stand_filter=DEFAULT,
                sit_stand_target=DEFAULT) as mocks:

            mock_sit_events = pd.DataFrame(
                data={
                    "time_step": [1, 1, 2, 2, 2, 2],
                    "disturbance_type_id": [
                        2, 1, 2, 2, 1, 1
                    ]
                })

            # since everything is thoroughly mocked, this data wont matter much
            # for the purposes of this test, other than the type is DataFrame,
            # and the lengths of the inventory array which is used to inform
            # the size of the output arrays
            mock_classifiers = pd.DataFrame({"c1": [1, 1, 1, 2, 2, 2]})
            mock_inventory = pd.DataFrame({"area": [5, 4, 3, 2, 1, 0]})
            mock_pools = pd.DataFrame({"pool1": [10, 20, 30, 40, 50, 60]})
            mock_state_variables = pd.DataFrame(
                {"land_class": [0, 1, 0, 1, 0, 1]})
            mock_params = pd.DataFrame(
                {"disturbance_type": [0]*mock_inventory.shape[0]})
            mock_cbm_vars = SimpleNamespace(
                classifiers=mock_classifiers,
                inventory=mock_inventory,
                pools=mock_pools,
                state=mock_state_variables,
                params=mock_params)
            # mock filter
            rule_filter = mocks["rule_filter"]

            rule_filter.create_filter = Mock()
            rule_filter.evaluate_filters = "mock_evaluate_filters"
            rule_filter.create_filter.side_effect = \
                lambda expression, data: "mock_filter"

            # mock rule target
            mocks["rule_target"].compute_disturbance_production = Mock()

            # mock event processor
            mock_event_processor = mocks["event_processor"]
            mock_event_processor.process_event = Mock()

            def mock_process_event(event_filters, undisturbed, target_func,
                                   disturbance_type_id, cbm_vars):
                call_count = mock_event_processor.process_event.call_count

                # using call count checks event sorting
                if call_count == 1:
                    self.assertTrue(disturbance_type_id == 1)
                else:
                    self.assertTrue(disturbance_type_id == 2)

                n_stands = cbm_vars.inventory.shape[0]
                self.assertTrue(list(undisturbed) == [True]*n_stands)
                self.assertTrue(target_func == "mock_target_factory")
                self.assertTrue(cbm_vars.classifiers.equals(mock_classifiers))
                self.assertTrue(cbm_vars.inventory.equals(mock_inventory))
                self.assertTrue(cbm_vars.pools.equals(mock_pools))
                self.assertTrue(
                    cbm_vars.state.equals(mock_state_variables))

                return SimpleNamespace(
                    cbm_vars=cbm_vars,
                    filter_result="mock_filter_result",
                    rule_target_result=SimpleNamespace(
                        statistics={"mock_stats": 1}
                    ))

            mock_event_processor.process_event.side_effect = mock_process_event

            # mock classifier filter
            mock_classifier_filter_builder = Mock()
            mock_classifier_filter_builder.create_classifiers_filter = Mock()

            # mock sit_stand_filter
            sit_stand_filter = mocks["sit_stand_filter"]
            sit_stand_filter.create_pool_filter_expression = Mock()
            sit_stand_filter.create_pool_filter_expression.side_effect = \
                lambda sit_event: "(mock_pool > 1)"
            sit_stand_filter.create_state_filter_expression = Mock()
            sit_stand_filter.create_state_filter_expression.side_effect = \
                lambda sit_event, age_only: "(mock_variable == 7)"
            sit_stand_filter.create_last_disturbance_type_filter = Mock()
            sit_stand_filter.create_last_disturbance_type_filter.side_effect \
                = lambda sit_event: ("", [])

            # mock sit_stand_target
            sit_stand_target = mocks["sit_stand_target"]
            sit_stand_target.create_sit_event_target_factory = Mock()
            sit_stand_target.create_sit_event_target_factory.side_effect = \
                lambda **kwargs: "mock_target_factory"

            # these mocks are not called by SITEventProcessor, but are
            # passed to underlying functions or objects
            mock_model_functions = "mock_model_functions"
            mock_compute_functions = "mock_compute_functions"
            mock_random_generator = "mock_random_generator"

            sit_event_processor = SITEventProcessor(
                model_functions=mock_model_functions,
                compute_functions=mock_compute_functions,
                classifier_filter_builder=mock_classifier_filter_builder,
                random_generator=mock_random_generator)

            cbm_vars_result, stats = sit_event_processor.process_events(
                time_step=1,  # there are 2 mock events with t = 1
                sit_events=mock_sit_events,
                cbm_vars=mock_cbm_vars)

            # all effects to these dataframes are mocked, so they should be
            # equal to the originals
            self.assertTrue(
                cbm_vars_result.classifiers.equals(mock_classifiers))
            self.assertTrue(cbm_vars_result.inventory.equals(mock_inventory))
            self.assertTrue(cbm_vars_result.pools.equals(mock_pools))
            self.assertTrue(
                cbm_vars_result.state.equals(mock_state_variables))
            self.assertTrue(
                cbm_vars_result.params.equals(mock_params))
            self.assertTrue(stats.shape[0] == 2)
