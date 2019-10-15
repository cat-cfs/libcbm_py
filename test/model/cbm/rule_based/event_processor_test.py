import unittest
from types import SimpleNamespace
import numpy as np
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

        mock_filter_data = SimpleNamespace(
            classifier_set="mock_classifier_set",
            pool_filter_expression="mock_pool_filter",
            pool_filter_columns="mock_pool_filter_columns",
            state_filter_expression="mock_tate_filter",
            state_filter_columns="mock_tate_filter_columns",
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

    def test_apply_filter_expected_result(self):

        mock_undisturbed = [False, True, False, True, True]

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

        mock_filter_factory = Mock()
        mock_filter_factory.merge_filters = Mock()

        def mock_merge_filters(*args):
            self.assertTrue(
                list(args) == [
                    "mock_created_filter", "mock_created_filter",
                    "mock_classifier_filter"])

        mock_filter_factory.merge_filters.side_effect = mock_merge_filters
        mock_filter_factory.create_filter = Mock()
        mock_filter_factory.create_filter.side_effect = \
            lambda expression, data, columns: "mock_created_filter"

        mock_filter_factory.evaluate_filter = Mock()
        mock_evaluate_filter_return = [False, True, False, True, False]
        mock_filter_factory.evaluate_filter.side_effect = \
            lambda merged_filter: mock_evaluate_filter_return

        mock_classifiers_filter_factory = Mock()

        def classifier_filter_factory(classifier_set, classifiers):
            return "mock_classifier_filter"
        mock_classifiers_filter_factory.side_effect = \
            lambda classifier_set, classifiers: "mock_classifier_filter"

        mock_filter_data = SimpleNamespace(
            classifier_set="mock_classifier_set",
            pool_filter_expression="mock_pool_filter",
            pool_filter_columns="mock_pool_filter_columns",
            state_filter_expression="mock_tate_filter",
            state_filter_columns="mock_tate_filter_columns",
        )

        result = event_processor.apply_filter(
            filter_factory=mock_filter_factory,
            classifiers_filter_factory=mock_classifiers_filter_factory,
            filter_data=mock_filter_data,
            undisturbed=mock_undisturbed,
            pools=mock_pools,
            state_variables=mock_state_variables,
            classifiers=mock_classifiers)
        self.assertTrue(
            list(result) == list(
                np.logical_and(mock_undisturbed, mock_evaluate_filter_return)))

    def test_apply_rule_based_event_expected_result(self):
        pass