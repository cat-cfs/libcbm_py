import unittest
from unittest.mock import patch
from unittest.mock import DEFAULT
import pandas as pd
from mock import Mock
from types import SimpleNamespace
from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import SITEventProcessor
from libcbm.model.cbm.rule_based.sit.sit_event_processor \
    import get_pre_dynamics_func

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.sit.sit_event_processor"


class SITEventProcessorTest(unittest.TestCase):

    def test_get_pre_dynamics_func(self):
        """tests get_pre_dynamics_func and also calls the function returned
        """
        with patch(PATCH_PATH + ".cbm_variables") as cbm_variables:

            time_step = 10
            cbm_variables.inventory_to_df = Mock()
            cbm_variables.inventory_to_df.side_effect = \
                lambda _: ("mock_classifiers", "mock_inventory")
            cbm_variables.initialize_cbm_parameters = Mock()
            cbm_variables.initialize_cbm_parameters.side_effect = \
                lambda **kwargs: "mock_params"
            cbm_variables.initialize_inventory = Mock()

            def mock_initialize_inventory(classifiers, inventory):
                self.assertTrue(classifiers == "mock_classifiers")
                self.assertTrue(inventory.equals(pd.DataFrame({"a": [1]})))
                return "mock_inventory_object_result"

            cbm_variables.initialize_inventory.side_effect = \
                mock_initialize_inventory
            cbm_variables.initialize_flux = Mock()
            cbm_variables.initialize_flux.side_effect = \
                lambda **kwargs: "mock_flux_result"

            cbm_vars = SimpleNamespace(
                inventory="mock_inventory_object",
                pools="mock_pools",
                state="mock_state",
                flux_indicators=pd.DataFrame({"a": [1], "b": [2], "c": [3]}))

            mock_sit_event_processor = Mock()
            mock_sit_event_processor.process_events = Mock()
            mock_sit_event_processor.process_events.side_effect = \
                lambda **kwargs: (
                    "mock_disturbance_types",
                    "mock_classifiers",
                    pd.DataFrame({"a": [1]}),  # .shape[0] is used internally
                    "mock_pools_result",
                    "mock_state_result"
                )

            mock_sit_events = "mock_events"
            pre_dynamics_func = get_pre_dynamics_func(
                mock_sit_event_processor, mock_sit_events)
            cbm_vars_result = pre_dynamics_func(time_step, cbm_vars)

            mock_sit_event_processor.process_events.assert_called_with(
                time_step=time_step,
                sit_events=mock_sit_events,
                classifiers="mock_classifiers",
                inventory="mock_inventory",
                pools="mock_pools",
                state_variables="mock_state")

            cbm_variables.inventory_to_df.assert_called_with(
                "mock_inventory_object")
            cbm_variables.initialize_cbm_parameters.assert_called_with(
                n_stands=1,
                disturbance_type="mock_disturbance_types"
            )
            cbm_variables.initialize_inventory.assert_called()
            cbm_variables.initialize_flux.assert_called_with(
                n_stands=1,
                flux_indicator_codes=["a", "b", "c"]
            )

            self.assertTrue(
                cbm_vars_result.inventory == "mock_inventory_object_result")
            self.assertTrue(
                cbm_vars_result.pools == "mock_pools_result")
            self.assertTrue(
                cbm_vars_result.state == "mock_state_result")
            self.assertTrue(
                cbm_vars_result.flux_indicators == "mock_flux_result")

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
                sit_stand_target=DEFAULT,
                cbm_variables=DEFAULT) as mocks:

            mock_sit_events = pd.DataFrame(
                data={
                    "time_step": [1, 1, 2, 2, 2, 2],
                    "disturbance_type_id": [
                        2, 1, 2, 2, 1, 1
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
