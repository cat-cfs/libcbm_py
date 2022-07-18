import unittest
from unittest.mock import patch
from types import SimpleNamespace
from mock import Mock
import pandas as pd
import numpy as np
from libcbm.storage import dataframe
from libcbm.storage import series
from libcbm.model.cbm.rule_based import event_processor

# used in patching (overriding) module imports in the module being tested
PATCH_PATH = "libcbm.model.cbm.rule_based.event_processor"


class EventProcessorTest(unittest.TestCase):
    def test_process_event_expected_result(self):
        """A test of the overall flow and a few of the internal calls
        of the process_event function.
        """

        with patch(PATCH_PATH + ".rule_filter") as mock_rule_filter:
            mock_pools = dataframe.from_pandas(
                pd.DataFrame(
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    columns=["a", "b", "c", "d"],
                )
            )

            mock_state_variables = dataframe.from_pandas(
                pd.DataFrame(
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    columns=["i", "j", "k"],
                )
            )

            mock_classifiers = dataframe.from_pandas(
                pd.DataFrame(
                    [[1, 1], [1, 1], [1, 1], [1, 1]], columns=["c1", "c2"]
                )
            )

            mock_inventory = dataframe.from_pandas(
                pd.DataFrame(
                    [[1, 1], [1, 1], [1, 1], [1, 1]], columns=["age", "area"]
                )
            )

            mock_params = dataframe.from_pandas(
                pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
            )

            mock_flux = dataframe.from_pandas(
                pd.DataFrame(
                    [[0, 0], [0, 0], [0, 0], [0, 0]], columns=["f1", "f2"]
                )
            )

            mock_cbm_vars = SimpleNamespace(
                classifiers=mock_classifiers,
                inventory=mock_inventory,
                state=mock_state_variables,
                pools=mock_pools,
                flux=mock_flux,
                parameters=mock_params,
            )
            disturbance_type_id = 5

            mock_evaluate_filter_return = series.from_list(
                "filter", [False, True, True, False]
            )
            mock_rule_filter.evaluate_filters = Mock()
            mock_rule_filter.evaluate_filters.side_effect = (
                lambda _: mock_evaluate_filter_return
            )

            mock_event_filters = ["mock_event_filter"]

            mock_undisturbed = series.from_list(
                "undisturbed", [True, True, True, True]
            )

            mock_target_func = Mock()

            def target_func(_, eligible):
                self.assertTrue(
                    eligible.to_list()
                    == (
                        mock_evaluate_filter_return & mock_undisturbed
                    ).to_list()
                )

                # mocks a disturbance target that fully disturbs inventory
                # records at index 1, 2
                return SimpleNamespace(
                    target=dataframe.from_pandas(
                        pd.DataFrame(
                            {
                                "disturbed_index": pd.Series([1, 2]),
                                "area_proportions": pd.Series([1.0, 1.0]),
                            }
                        )
                    ),
                    statistics="mock_statistics",
                )

            mock_target_func.side_effect = target_func

            res = event_processor.process_event(
                event_filters=mock_event_filters,
                undisturbed=mock_undisturbed,
                target_func=mock_target_func,
                disturbance_type_id=disturbance_type_id,
                cbm_vars=mock_cbm_vars,
            )

            mock_rule_filter.evaluate_filters.assert_called_once_with(
                *mock_event_filters
            )
            mock_target_func.assert_called_once()

            # no splits occurred here, so the inputs are returned
            self.assertTrue(
                res.cbm_vars.classifiers.to_pandas().equals(
                    mock_classifiers.to_pandas()
                )
            )
            self.assertTrue(
                res.cbm_vars.inventory.to_pandas().equals(
                    mock_inventory.to_pandas()
                )
            )
            self.assertTrue(
                res.cbm_vars.pools.to_pandas().equals(mock_pools.to_pandas())
            )
            self.assertTrue(
                res.cbm_vars.state.to_pandas().equals(
                    mock_state_variables.to_pandas()
                )
            )
            self.assertTrue(
                res.cbm_vars.flux.to_pandas().equals(mock_flux.to_pandas())
            )
            self.assertTrue(
                res.cbm_vars.parameters.to_pandas().equals(
                    mock_params.to_pandas()
                )
            )

    def test_apply_rule_based_event_expected_result_with_no_split(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=dataframe.from_pandas(
                pd.DataFrame({"classifier1": [1, 2, 3, 4]})
            ),
            inventory=dataframe.from_pandas(
                pd.DataFrame({"area": [1, 2, 3, 4]})
            ),
            state=dataframe.from_pandas(pd.DataFrame({"s1": [1, 2, 3, 4]})),
            pools=dataframe.from_pandas(pd.DataFrame({"p1": [1, 2, 3, 4]})),
            flux=dataframe.from_pandas(pd.DataFrame({"f1": [1, 2, 3, 4]})),
            parameters=dataframe.from_pandas(
                pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
            ),
        )
        disturbance_type_id = 11
        cbm_vars = event_processor.apply_rule_based_event(
            target=dataframe.from_pandas(
                pd.DataFrame(
                    {
                        "disturbed_index": pd.Series([1, 2]),
                        "area_proportions": pd.Series([1.0, 1.0]),
                    }
                )
            ),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars,
        )

        self.assertTrue(
            cbm_vars.classifiers.to_pandas().equals(
                pd.DataFrame({"classifier1": [1, 2, 3, 4]})
            )
        )
        self.assertTrue(
            cbm_vars.inventory.to_pandas().equals(
                pd.DataFrame({"area": [1, 2, 3, 4]})
            )
        )
        self.assertTrue(
            cbm_vars.state.to_pandas().equals(
                pd.DataFrame({"s1": [1, 2, 3, 4]})
            )
        )
        self.assertTrue(
            cbm_vars.pools.to_pandas().equals(
                pd.DataFrame({"p1": [1, 2, 3, 4]})
            )
        )
        self.assertTrue(
            cbm_vars.flux.to_pandas().equals(
                pd.DataFrame({"f1": [1, 2, 3, 4]})
            )
        )
        self.assertTrue(
            cbm_vars.parameters.to_pandas().equals(
                pd.DataFrame({"disturbance_type": [0, 11, 11, 0]})
            )
        )

    def test_apply_rule_based_event_expected_result_with_split(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=dataframe.from_pandas(
                pd.DataFrame({"classifier1": [1, 2, 3, 4]})
            ),
            inventory=dataframe.from_pandas(
                pd.DataFrame({"area": [1, 2, 3, 4]})
            ),
            state=dataframe.from_pandas(pd.DataFrame({"s1": [1, 2, 3, 4]})),
            pools=dataframe.from_pandas(pd.DataFrame({"p1": [1, 2, 3, 4]})),
            flux=dataframe.from_pandas(pd.DataFrame({"f1": [1, 2, 3, 4]})),
            parameters=dataframe.from_pandas(
                pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
            ),
        )

        disturbance_type_id = 9000
        cbm_vars_result = event_processor.apply_rule_based_event(
            target=dataframe.from_pandas(
                pd.DataFrame(
                    {
                        "disturbed_index": pd.Series([0, 1, 2]),
                        "area_proportions": pd.Series([1.0, 0.85, 0.9]),
                    }
                )
            ),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars,
        )

        self.assertTrue(
            cbm_vars_result.classifiers.to_pandas().equals(
                pd.DataFrame({"classifier1": [1, 2, 3, 4, 2, 3]})
            )
        )
        self.assertTrue(
            np.allclose(
                cbm_vars_result.inventory["area"].to_numpy(),
                [
                    1,
                    2 * 0.85,  # index=1 is split at 0.85
                    3 * 0.9,  # index=2 is split at 0.9
                    4,
                    2 * 0.15,  # the remainder of index=1
                    3 * 0.1,  # the remainder of index=2
                ],
            )
        )

        self.assertTrue(
            cbm_vars_result.pools.to_pandas().equals(
                pd.DataFrame({"p1": [1, 2, 3, 4, 2, 3]})
            )
        )
        self.assertTrue(
            cbm_vars_result.state.to_pandas().equals(
                pd.DataFrame({"s1": [1, 2, 3, 4, 2, 3]})
            )
        )
        self.assertTrue(
            cbm_vars_result.flux.to_pandas().equals(
                pd.DataFrame({"f1": [1, 2, 3, 4, 2, 3]})
            )
        )
        self.assertTrue(
            cbm_vars_result.parameters.to_pandas().equals(
                pd.DataFrame({"disturbance_type": [9000, 9000, 9000, 0, 0, 0]})
            )
        )

    def test_apply_rule_based_event_expected_result_no_target_rows(self):

        mock_cbm_vars = SimpleNamespace(
            classifiers=dataframe.from_pandas(
                pd.DataFrame({"classifier1": [1, 2, 3, 4]})
            ),
            inventory=dataframe.from_pandas(
                pd.DataFrame({"area": [1, 2, 3, 4]})
            ),
            state=dataframe.from_pandas(pd.DataFrame({"s1": [1, 2, 3, 4]})),
            pools=dataframe.from_pandas(pd.DataFrame({"p1": [1, 2, 3, 4]})),
            flux=dataframe.from_pandas(pd.DataFrame({"f1": [1, 2, 3, 4]})),
            parameters=dataframe.from_pandas(
                pd.DataFrame({"disturbance_type": [0, 0, 0, 0]})
            ),
        )

        disturbance_type_id = 9000
        cbm_vars_result = event_processor.apply_rule_based_event(
            target=dataframe.from_pandas(
                pd.DataFrame(
                    {
                        "disturbed_index": pd.Series(dtype=int),
                        "area_proportions": pd.Series(dtype=float),
                    }
                )
            ),
            disturbance_type_id=disturbance_type_id,
            cbm_vars=mock_cbm_vars,
        )

        self.assertTrue(
            cbm_vars_result.classifiers.to_pandas().equals(
                mock_cbm_vars.classifiers.to_pandas()
            )
        )
        self.assertTrue(
            cbm_vars_result.inventory.to_pandas().equals(
                mock_cbm_vars.inventory.to_pandas()
            )
        )
        self.assertTrue(
            cbm_vars_result.state.to_pandas().equals(
                mock_cbm_vars.state.to_pandas()
            )
        )
        self.assertTrue(
            cbm_vars_result.pools.to_pandas().equals(
                mock_cbm_vars.pools.to_pandas()
            )
        )
        self.assertTrue(
            cbm_vars_result.flux.to_pandas().equals(
                mock_cbm_vars.flux.to_pandas()
            )
        )
        self.assertTrue(
            cbm_vars_result.parameters.to_pandas().equals(
                mock_cbm_vars.parameters.to_pandas()
            )
        )
