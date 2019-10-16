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

        mock_evaluate_filter_return = [False, True, True, False]
        mock_evaluate_filter = Mock()
        mock_evaluate_filter.side_effect = \
            lambda _: mock_evaluate_filter_return

        mock_event_filter = "mock_event_filter"

        mock_undisturbed = [True, True, True, True]

        mock_target_func = Mock()

        def target_func(pool, inventory, state):
            event_filter = np.logical_and(
                mock_evaluate_filter_return, mock_undisturbed)

            self.assertTrue(pool.equals(mock_pools[event_filter]))
            self.assertTrue(inventory.equals(mock_inventory[event_filter]))
            self.assertTrue(state.equals(mock_state_variables[event_filter]))
            # mocks a disturbance target that fully disturbs inventory records
            # at index 1, 2
            return {
                "disturbed_index": pd.Series([1, 2]),
                "area_proportions": pd.Series([1.0, 1.0])
            }

        mock_target_func.side_effect = target_func

        target, classifiers, inventory, pools, state_variables = \
            event_processor.process_event(
                filter_evaluator=mock_evaluate_filter,
                event_filter=mock_event_filter,
                undisturbed=mock_undisturbed,
                target_func=mock_target_func,
                classifiers=mock_classifiers,
                inventory=mock_inventory,
                pools=mock_pools,
                state_variables=mock_state_variables)

        mock_evaluate_filter.assert_called_once_with("mock_event_filter")
        mock_target_func.assert_called_once()

        self.assertTrue(target["disturbed_index"].equals(pd.Series([1, 2])))
        self.assertTrue(target["area_proportions"].equals(pd.Series([1.0, 1.0])))
        # no splits occurred here, so the inputs are returned
        self.assertTrue(classifiers.equals(mock_classifiers))
        self.assertTrue(inventory.equals(mock_inventory))
        self.assertTrue(pools.equals(mock_pools))
        self.assertTrue(state_variables.equals(mock_state_variables))

    def test_apply_rule_based_event_expected_result_with_no_split(self):

        target, classifiers, inventory, pools, state_variables = \
            event_processor.apply_rule_based_event(
                target=pd.DataFrame({
                    "disturbed_index": pd.Series([1, 2]),
                    "area_proportions": pd.Series([1.0, 1.0])}),
                classifiers=pd.DataFrame({"classifier1": [1, 2, 3, 4]}),
                inventory=pd.DataFrame({"area": [1, 2, 3, 4]}),
                pools=pd.DataFrame({"p1": [1, 2, 3, 4]}),
                state_variables=pd.DataFrame({"s1": [1, 2, 3, 4]}))

        self.assertTrue(target["disturbed_index"].equals(pd.Series([1, 2])))
        self.assertTrue(target["area_proportions"].equals(pd.Series([1.0, 1.0])))

        self.assertTrue(
            classifiers.equals(pd.DataFrame({"classifier1": [1, 2, 3, 4]})))
        self.assertTrue(
            inventory.equals(pd.DataFrame({"area": [1, 2, 3, 4]})))
        self.assertTrue(
            pools.equals(pd.DataFrame({"p1": [1, 2, 3, 4]})))
        self.assertTrue(
            state_variables.equals(pd.DataFrame({"s1": [1, 2, 3, 4]})))

    def test_apply_rule_based_event_expected_result_with_split(self):

        target, classifiers, inventory, pools, state_variables = \
            event_processor.apply_rule_based_event(
                target=pd.DataFrame({
                    "disturbed_index": pd.Series([0, 1, 2]),
                    "area_proportions": pd.Series([1.0, 0.85, 0.9])}),
                classifiers=pd.DataFrame({"classifier1": [1, 2, 3, 4]}),
                inventory=pd.DataFrame({"area": [1, 2, 3, 4]}),
                pools=pd.DataFrame({"p1": [1, 2, 3, 4]}),
                state_variables=pd.DataFrame({"s1": [1, 2, 3, 4]}))

        self.assertTrue(target["disturbed_index"].equals(pd.Series([0, 1, 2])))
        self.assertTrue(target["area_proportions"].equals(pd.Series([1.0, 0.85, 0.9])))

        self.assertTrue(
            classifiers.equals(pd.DataFrame(
                {"classifier1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            np.allclose(
                inventory.area,
                [
                    1,
                    2 * 0.85,  # index=1 is split at 0.85
                    3 * 0.9,   # index=2 is split at 0.9
                    4,
                    2 * 0.15,  # the remainder of index=1
                    3 * 0.1    # the remainder of index=2
                ]))
        self.assertTrue(
            pools.equals(pd.DataFrame({"p1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            state_variables.equals(pd.DataFrame({"s1": [1, 2, 3, 4, 2, 3]})))
