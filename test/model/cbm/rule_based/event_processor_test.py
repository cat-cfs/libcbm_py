import unittest
from types import SimpleNamespace
import pandas as pd
from mock import Mock
from libcbm.model.cbm.rule_based import event_processor


class EventProcessorTest(unittest.TestCase):

    def test_process_event_expected_result(self):
        """A test of the overall flow and a few of the internal calls
        of the process_event function.
        """
        mock_filter_factory = Mock()

        mock_filter_factory.merge_filters = Mock()

        mock_filter_factory.create_filter = Mock()

        mock_filter_factory.evaluate_filter = Mock()
        mock_filter_factory.evaluate_filter.side_effect = \
            lambda _: [False, True, True, False]

        classifiers_filter_factory = Mock()
        classifiers_filter_factory.filter_data = Mock()

        mock_filter_data = SimpleNamespace(
            classifier_set="mock_classifier_set",
            pool_filter_expression="mock_pool_filter",
            pool_filter_columns="",
            state_filter_expression="",
            state_filter_columns="",
        )

        mock_undisturbed = [True, True, True, True]

        mock_target_func = Mock()
        mock_target_func.side_effect = lambda a, b, c: {
            # mocks a disturbance target that fully disturbs inventory records
            # at index 1, 2
            "disturbed_index": pd.Series([1, 2]),
            "area_proportions": pd.Series([1.0, 1.0])
        }

        mock_pools = pd.DataFrame([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], columns=["a", "b", "c", "d"])

        mock_state_variables = pd.DataFrame([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], columns=["i", "j", "k"])

        mock_classifiers = pd.DataFrame([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ], columns=["c1", "c2"])

        mock_inventory = pd.DataFrame([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ], columns=["age", "area"])

        index, inventory, classifiers, state_variables = \
            event_processor.process_event(
                filter_factory=mock_filter_factory,
                classifiers_filter_factory=classifiers_filter_factory,
                filter_data=mock_filter_data,
                undisturbed=mock_undisturbed,
                target_func=mock_target_func,
                pools=mock_pools,
                state_variables=mock_state_variables,
                classifiers=mock_classifiers,
                inventory=mock_inventory)
        self.assertTrue(list(index) == [1, 2])
        # no splits occurred here, so the inputs are returned
        self.assertTrue(inventory.equals(mock_inventory))
        self.assertTrue(classifiers.equals(mock_classifiers))
        self.assertTrue(state_variables.equals(mock_state_variables))
