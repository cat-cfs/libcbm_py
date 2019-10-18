import unittest
from unittest.mock import patch
from unittest.mock import DEFAULT
import pandas as pd
from mock import Mock

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
                    "time_step": [1, 1, 2, 2, 2, 2],
                    "disturbance_type": [
                        "harvest", "fire", "harvest", "harvest", "fire", "fire"
                    ]
                })

            # Going to model 2 events by setting time_step = 1, which
            # corresponds to the the first 2 rows of the above mock_sit_events
            # df.

            # event 1 on timestep 1 will use this target, meaning:
            #   - the index 0 is targeted to be fully disturbed
            #   - the index 1 is targeted to be split in half
            mock_target_1 = {
                "area_proportions": pd.Series([1.0, 0.5]),
                "disturbed_index": pd.Series([0, 1])
            }
            # event 2 on timestep 1 will use this target, meaning:
            #   - the index 2 is targeted to be fully disturbed
            #   - the index 3 is targeted to be split in half
            mock_target_2 = {
                "area_proportions": pd.Series([1.0, 0.5]),
                "disturbed_index": pd.Series([2, 3])
            }

            # since everything is thoroughly mocked, this data wont matter much
            # for the purposes of this test, other than the type is DataFrame,
            # and the lengths of the inventory array which is used to inform
            # the size of the output arrays
            mock_classifiers = pd.DataFrame({"c1": [1, 1, 1, 2, 2, 2]})
            mock_inventory = pd.DataFrame({"area": [5, 4, 3, 2, 1, 0]})
            mock_pools = pd.DataFrame({"pool1": [10, 20, 30, 40, 50, 60]})
            mock_state_variables = pd.DataFrame(
                {"land_class": [0, 1, 0, 1, 0, 1]})

            # mock filter
            rule_filter = mocks["rule_filter"]
            rule_filter.merge_filters = Mock()
            rule_filter.merge_filters.side_effect = \
                lambda *filters: "mock_merged_filter"
            rule_filter.create_filter = Mock()
            rule_filter.evaluate_filter = "mock_evaluate_filter"
            rule_filter.create_filter.side_effect = \
                lambda expression, data, columns: "mock_filter"

            # mock rule target
            mocks["rule_target"].compute_disturbance_production = Mock()

            # mock event processor
            mock_event_processor = mocks["event_processor"]
            mock_event_processor.process_event = Mock()

            def mock_process_event(filter_evaluator, event_filter,
                                   undisturbed, target_func, classifiers,
                                   inventory, pools, state_variables):

                call_count = mock_event_processor.process_event.call_count
                # expecting 2 calls to this function, one for each of the
                # events on timestep one in the mock sit events
                mock_target = None
                if call_count == 1:
                    mock_target = mock_target_1
                    # all records are undisturbed on first call
                    self.assertTrue(
                        list(undisturbed) == [1, 1, 1, 1, 1, 1])
                elif call_count == 2:
                    mock_target = mock_target_2
                    # as of the second call to this function, the first 2
                    # indices have been targeted for disturbance, and an
                    # additional item has been added to the array to represent
                    # the split record
                    self.assertTrue(
                        list(undisturbed) == [0, 0, 1, 1, 1, 1, 1])

                self.assertTrue(filter_evaluator == "mock_evaluate_filter")
                self.assertTrue(event_filter == "mock_merged_filter")
                self.assertTrue(target_func == "mock_target_factory")
                self.assertTrue(classifiers.equals(mock_classifiers))
                self.assertTrue(inventory.equals(mock_inventory))
                self.assertTrue(pools.equals(mock_pools))
                self.assertTrue(state_variables.equals(state_variables))

                return (
                    mock_target, classifiers, inventory, pools,
                    state_variables)

            mock_event_processor.process_event.side_effect = mock_process_event

            # mock cbm defaults reference
            mock_cbm_defaults_ref = Mock()
            mock_cbm_defaults_ref.get_disturbance_types = Mock()
            mock_cbm_defaults_ref.get_disturbance_types.side_effect = \
                lambda: [
                    {"disturbance_type_name": "fire",
                     "disturbance_type_id": 1},
                    {"disturbance_type_name": "harvest",
                     "disturbance_type_id": 2}]

            mock_cbm_defaults_ref.get_flux_indicators = Mock()
            mock_cbm_defaults_ref.get_flux_indicators.side_effect = \
                lambda: ["a", "b", "c"]

            # mock classifier filter
            mock_classifier_filter_builder = Mock()
            mock_classifier_filter_builder.create_classifiers_filter = Mock()

            # mock sit_stand_filter
            sit_stand_filter = mocks["sit_stand_filter"]
            sit_stand_filter.create_pool_filter_expression = Mock()
            sit_stand_filter.create_pool_filter_expression.side_effect = \
                lambda sit_event: ("(mock_pool > 1)", ["mock_pool"])
            sit_stand_filter.create_state_filter_expression = Mock()
            sit_stand_filter.create_state_filter_expression.side_effect = \
                lambda sit_event, age_only: \
                ("(mock_variable == 7)", ["mock_variable"])

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
                    time_step=1,  # there are 2 mock events with t = 1
                    sit_events=mock_sit_events,
                    classifiers=mock_classifiers,
                    inventory=mock_inventory,
                    pools=mock_pools,
                    state_variables=mock_state_variables)

            # all effects to these dataframes are mocked, so they should be
            # equal to the originals
            self.assertTrue(classifiers.equals(mock_classifiers))
            self.assertTrue(inventory.equals(mock_inventory))
            self.assertTrue(pools.equals(mock_pools))
            self.assertTrue(state_variables.equals(state_variables))

            # note: the sort order took effect meaning the fire id was applied
            # on the first iteration, and the harvest id on the second.
            self.assertTrue(list(dist_types) == [1, 1, 2, 2, 0, 0, 0, 0])
