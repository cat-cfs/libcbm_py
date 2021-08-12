import unittest
from unittest.mock import patch
from types import SimpleNamespace
from mock import Mock
import numpy as np
import pandas as pd

from libcbm.model.cbm.rule_based import event_processor

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.event_processor"


class EventProcessorTest(unittest.TestCase):

    def test_process_event_expected_result(self):
        """A test of the overall flow and a few of the internal calls
        of the process_event function.
        """

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
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

            mock_params = pd.DataFrame(
                {"disturbance_type": [0, 0, 0, 0]})

            mock_flux_indicators = pd.DataFrame([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]
            ], columns=["f1", "f2"])

            mock_cbm_vars = SimpleNamespace(
                classifiers=mock_classifiers,
                inventory=mock_inventory,
                state=mock_state_variables,
                pools=mock_pools,
                flux_indicators=mock_flux_indicators,
                params=mock_params
            )
            disturbance_type_id = 5

            mock_evaluate_filter_return = \
                [False, True, True, False]
            mock_rule_filter.evaluate_filter = Mock()
            mock_rule_filter.evaluate_filter.side_effect = \
                lambda _: mock_evaluate_filter_return

            mock_event_filter = "mock_event_filter"

            mock_undisturbed = [True, True, True, True]

            mock_target_func = Mock()

            def target_func(_, eligible):
                self.assertTrue(list(eligible) == list(np.logical_and(
                    mock_evaluate_filter_return, mock_undisturbed)))
                # mocks a disturbance target that fully disturbs inventory
                # records at index 1, 2
                return SimpleNamespace(
                    target={
                        "disturbed_index": pd.Series([1, 2]),
                        "area_proportions": pd.Series([1.0, 1.0])
                    },
                    statistics="mock_statistics")

            mock_target_func.side_effect = target_func

            res = event_processor.process_event(
                event_filter=mock_event_filter,
                undisturbed=mock_undisturbed,
                target_func=mock_target_func,
                disturbance_type_id=disturbance_type_id,
                cbm_vars=mock_cbm_vars)

            mock_rule_filter.evaluate_filter.assert_called_once_with(
                "mock_event_filter")
            mock_target_func.assert_called_once()

            # no splits occurred here, so the inputs are returned
            self.assertTrue(
                res.cbm_vars.classifiers.equals(mock_classifiers))
            self.assertTrue(
                res.cbm_vars.inventory.equals(mock_inventory))
            self.assertTrue(
                res.cbm_vars.pools.equals(mock_pools))
            self.assertTrue(
                res.cbm_vars.state.equals(mock_state_variables))
            self.assertTrue(
                res.cbm_vars.flux_indicators.equals(mock_flux_indicators))
            self.assertTrue(
                res.cbm_vars.params.equals(mock_params))

    def test_apply_rule_based_event_expected_result_with_no_split(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=pd.DataFrame({"classifier1": [1, 2, 3, 4]}),
            inventory=pd.DataFrame({"area": [1, 2, 3, 4]}),
            state=pd.DataFrame({"s1": [1, 2, 3, 4]}),
            pools=pd.DataFrame({"p1": [1, 2, 3, 4]}),
            flux_indicators=pd.DataFrame({"f1": [1, 2, 3, 4]}),
            params=pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
        )
        disturbance_type_id = 11
        cbm_vars = event_processor.apply_rule_based_event(
            target=pd.DataFrame({
                "disturbed_index": pd.Series([1, 2]),
                "area_proportions": pd.Series([1.0, 1.0])}),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars)

        self.assertTrue(
            cbm_vars.classifiers.equals(
                pd.DataFrame({"classifier1": [1, 2, 3, 4]})))
        self.assertTrue(
            cbm_vars.inventory.equals(
                pd.DataFrame({"area": [1, 2, 3, 4]})))
        self.assertTrue(
            cbm_vars.state.equals(
                pd.DataFrame({"s1": [1, 2, 3, 4]})))
        self.assertTrue(
            cbm_vars.pools.equals(
                pd.DataFrame({"p1": [1, 2, 3, 4]})))
        self.assertTrue(
            cbm_vars.flux_indicators.equals(
                pd.DataFrame({"f1": [1, 2, 3, 4]})))
        self.assertTrue(
            cbm_vars.params.equals(
                pd.DataFrame({"disturbance_type": [0, 11, 11, 0]})))

    def test_apply_rule_based_event_expected_result_with_split(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=pd.DataFrame({"classifier1": [1, 2, 3, 4]}),
            inventory=pd.DataFrame({"area": [1, 2, 3, 4]}),
            state=pd.DataFrame({"s1": [1, 2, 3, 4]}),
            pools=pd.DataFrame({"p1": [1, 2, 3, 4]}),
            flux_indicators=pd.DataFrame({"f1": [1, 2, 3, 4]}),
            params=pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
        )

        disturbance_type_id = 9000
        cbm_vars_result = event_processor.apply_rule_based_event(
            target=pd.DataFrame({
                "disturbed_index": pd.Series([0, 1, 2]),
                "area_proportions": pd.Series([1.0, 0.85, 0.9])}),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars)

        self.assertTrue(
            cbm_vars_result.classifiers.equals(pd.DataFrame(
                {"classifier1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            np.allclose(
                cbm_vars_result.inventory.area,
                [
                    1,
                    2 * 0.85,  # index=1 is split at 0.85
                    3 * 0.9,   # index=2 is split at 0.9
                    4,
                    2 * 0.15,  # the remainder of index=1
                    3 * 0.1    # the remainder of index=2
                ]))

        self.assertTrue(
            cbm_vars_result.pools.equals(
                pd.DataFrame({"p1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            cbm_vars_result.state.equals(
                pd.DataFrame({"s1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            cbm_vars_result.flux_indicators.equals(
                pd.DataFrame({"f1": [1, 2, 3, 4, 2, 3]})))
        self.assertTrue(
            cbm_vars_result.params.equals(
                pd.DataFrame(
                    {"disturbance_type": [
                        9000, 9000, 9000, 0, 0, 0]})))

    def test_apply_rule_based_event_expected_result_no_target_rows(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=pd.DataFrame({"classifier1": [1, 2, 3, 4]}),
            inventory=pd.DataFrame({"area": [1, 2, 3, 4]}),
            state=pd.DataFrame({"s1": [1, 2, 3, 4]}),
            pools=pd.DataFrame({"p1": [1, 2, 3, 4]}),
            flux_indicators=pd.DataFrame({"f1": [1, 2, 3, 4]}),
            params=pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
        )

        disturbance_type_id = 9000
        cbm_vars_result = event_processor.apply_rule_based_event(
            target=pd.DataFrame({
                "disturbed_index": pd.Series(dtype=int),
                "area_proportions": pd.Series(dtype=float)}),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars)

        self.assertTrue(
            cbm_vars_result.classifiers.equals(mock_cbm_vars.classifiers))
        self.assertTrue(
            cbm_vars_result.inventory.equals(mock_cbm_vars.inventory))
        self.assertTrue(
            cbm_vars_result.state.equals(mock_cbm_vars.state))
        self.assertTrue(
            cbm_vars_result.pools.equals(mock_cbm_vars.pools))
        self.assertTrue(
            cbm_vars_result.flux_indicators.equals(
                mock_cbm_vars.flux_indicators))
        self.assertTrue(
            cbm_vars_result.params.equals(mock_cbm_vars.params))
