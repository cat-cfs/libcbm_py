import unittest
import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based import rule_target
from libcbm.storage import series
from libcbm.storage import dataframe


class RuleTargetTest(unittest.TestCase):
    def test_sorted_disturbance_target_error_on_less_than_zero_target(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=series.from_pandas(pd.Series([10, 1000, 0])),
                sort_var=series.from_pandas(pd.Series([1, 2, 3])),
                target=-10,
                eligible=series.from_pandas(pd.Series([True, True, True])),
            )

    def test_sorted_disturbance_target_error_on_lt_zero_target_var(self):
        with self.assertRaises(ValueError):
            rule_target.sorted_disturbance_target(
                target_var=series.from_pandas(pd.Series([-1, 1000, 0])),
                sort_var=series.from_pandas(pd.Series([1, 2, 3])),
                target=10,
                eligible=series.from_pandas(pd.Series([True, True, True])),
            )

    def test_sorted_disturbance_target_unrealized_on_zero_target_var_sum(self):

        result = rule_target.sorted_disturbance_target(
            target_var=series.from_pandas(pd.Series([0, 0, 0])),
            sort_var=series.from_pandas(pd.Series([1, 2, 3])),
            target=10,
            eligible=series.from_pandas(pd.Series([True, True, True])),
        )
        self.assertTrue(result.target is None)

        self.assertTrue(result.statistics["total_eligible_value"] == 0)
        self.assertTrue(result.statistics["total_achieved"] == 0)
        self.assertTrue(result.statistics["shortfall"] == 10)
        self.assertTrue(result.statistics["num_records_disturbed"] == 0)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_sorted_disturbance_target_on_unrealized_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=series.from_pandas(pd.Series([33, 33, 33])),
            sort_var=series.from_pandas(pd.Series([1, 2, 3])),
            target=100,
            eligible=series.from_pandas(pd.Series([True, True, True])),
        )
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [2, 1, 0]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list() == [1.0, 1.0, 1.0]
        )
        self.assertTrue(result.target["target_var"].to_list() == [33, 33, 33])
        self.assertTrue(result.target["sort_var"].to_list() == [3, 2, 1])

        self.assertTrue(result.statistics["total_eligible_value"] == 33 * 3)
        self.assertTrue(result.statistics["total_achieved"] == 33 * 3)
        self.assertTrue(result.statistics["shortfall"] == 100 - 33 * 3)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_sorted_disturbance_target_expected_result_with_exact_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=series.from_pandas(pd.Series([33, 33, 33])),
            sort_var=series.from_pandas(pd.Series([1, 2, 3])),
            target=99,
            eligible=series.from_pandas(pd.Series([True, True, True])),
        )
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [2, 1, 0]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list() == [1.0, 1.0, 1.0]
        )
        self.assertTrue(result.target["target_var"].to_list() == [33, 33, 33])
        self.assertTrue(result.target["sort_var"].to_list() == [3, 2, 1])

        self.assertTrue(result.statistics["total_eligible_value"] == 33 * 3)
        self.assertTrue(result.statistics["total_achieved"] == 33 * 3)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_sorted_disturbance_target_expected_result_with_less_target(self):

        result = rule_target.sorted_disturbance_target(
            target_var=series.from_pandas(pd.Series([33, 33, 33])),
            sort_var=series.from_pandas(pd.Series([1, 2, 3])),
            target=34,
            eligible=series.from_pandas(pd.Series([True, True, True])),
        )
        # cbm sorts descending for disturbance targets (oldest first, etc.)
        self.assertTrue(result.target["disturbed_index"].to_list() == [2, 1])
        self.assertTrue(
            result.target["area_proportions"].to_list() == [1.0, 1 / 33]
        )
        self.assertTrue(result.target["target_var"].to_list() == [33, 33])
        self.assertTrue(result.target["sort_var"].to_list() == [3, 2])

        self.assertTrue(result.statistics["total_eligible_value"] == 33 * 3)
        self.assertTrue(result.statistics["total_achieved"] == 34)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 2)
        self.assertTrue(result.statistics["num_splits"] == 1)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_sorted_area_target_expected_result(self):

        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [1.5, 2.0, 2.0, 3.0]}
            )
        )
        result = rule_target.sorted_area_target(
            area_target_value=5.1,
            sort_value=mock_inventory["age"],
            inventory=mock_inventory,
            eligible=series.from_pandas(pd.Series([True, True, True, True])),
        )
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [3, 1, 2]
        )
        self.assertTrue(
            result.target["target_var"].to_list() == [3.0, 2.0, 2.0]
        )
        self.assertTrue(result.target["sort_var"].to_list() == [30, 20, 10])
        self.assertTrue(
            np.allclose(
                result.target["area_proportions"].to_numpy(),
                [1.0, 1.0, 0.1 / 2.0],
            )
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == mock_inventory["area"].sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 5.1)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 1)
        self.assertTrue(result.statistics["num_eligible"] == 4)

    def test_sorted_area_target_error_on_dimension_mismatch(self):

        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [1.5, 2.0, 2.0, 3.0]}
            )
        )
        # note only 3 sort values, with 4 inventory rows
        with self.assertRaises(ValueError):
            rule_target.sorted_area_target(
                area_target_value=5.1,
                sort_value=series.from_pandas(pd.Series([1, 2, 3])),
                inventory=mock_inventory,
                eligible=series.from_pandas(
                    pd.Series([True, True, True, True])
                ),
            )

    def test_sorted_merch_target_no_eligible_C(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [2.0, 2.0, 2.0, 2.0]}
            )
        )
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10]})
        )  # tonnes C/ha
        result = rule_target.sorted_merch_target(
            carbon_target=55,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=series.from_pandas(pd.Series([4, 3, 2, 1])),
            efficiency=1.0,
            eligible=series.from_pandas(
                pd.Series([False, False, False, False])
            ),
        )
        self.assertTrue(result.target is None)
        self.assertTrue(
            result.statistics
            == {
                "total_eligible_value": 0,
                "total_achieved": 0,
                "shortfall": 55,
                "num_records_disturbed": 0,
                "num_splits": 0,
                "num_eligible": 0,
            }
        )

    def test_sorted_merch_target_expected_result(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [2.0, 2.0, 2.0, 2.0]}
            )
        )
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10]})
        )  # tonnes C/ha
        # since C targets are accumulated on mass values
        # the total production values here are actually
        # 20,20,20,20 tonnes using the above area multipliers

        result = rule_target.sorted_merch_target(
            carbon_target=55,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=series.from_pandas(pd.Series([4, 3, 2, 1])),
            efficiency=1.0,
            eligible=series.from_pandas(pd.Series([True, True, True, True])),
        )
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 1, 2]
        )
        self.assertTrue(result.target["target_var"].to_list() == [20, 20, 20])
        self.assertTrue(result.target["sort_var"].to_list() == [4, 3, 2])
        self.assertTrue(
            np.allclose(
                result.target["area_proportions"].to_numpy(),
                [1.0, 1.0, 15 / 20],
            )
        )
        self.assertTrue(
            result.statistics["total_eligible_value"]
            == (
                mock_inventory["area"] * mock_disturbance_production["Total"]
            ).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 55)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 1)
        self.assertTrue(result.statistics["num_eligible"] == 4)

    def test_sorted_merch_target_expected_result_unrealized(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [2.0, 2.0, 2.0, 2.0]}
            )
        )
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10]})
        )  # tonnes C/ha
        # since C targets are accumulated on mass values
        # the total production values here are actually
        # 20,20,20,20 tonnes using the above area multipliers

        # since the last index is not eligible maning the total production
        # available is 60 tonnes, and the target is 65
        result = rule_target.sorted_merch_target(
            carbon_target=65,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=series.from_pandas(pd.Series([4, 3, 2, 1])),
            efficiency=1.0,
            eligible=series.from_pandas(pd.Series([True, True, True, False])),
        )  # note ineligible

        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 1, 2]
        )
        self.assertTrue(result.target["target_var"].to_list() == [20, 20, 20])
        self.assertTrue(result.target["sort_var"].to_list() == [4, 3, 2])
        self.assertTrue(
            np.allclose(
                result.target["area_proportions"].to_numpy(), [1.0, 1.0, 1.0]
            )
        )
        self.assertTrue(result.statistics["total_eligible_value"] == 60)
        self.assertTrue(result.statistics["total_achieved"] == 60)
        self.assertTrue(result.statistics["shortfall"] == 5)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_sorted_merch_target_error_on_dimension_mismatch1(self):

        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [2.0, 2.0, 2.0, 2.0]}
            )
        )

        # note 5 values here, and 4 in inventory
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10, 10]})
        )

        with self.assertRaises(ValueError):
            rule_target.sorted_merch_target(
                carbon_target=55,
                disturbance_production=mock_disturbance_production,
                inventory=mock_inventory,
                sort_value=series.from_pandas(pd.Series([4, 3, 2, 1])),
                efficiency=1.0,
                eligible=series.from_pandas(
                    pd.Series([True, True, True, True])
                ),
            )

    def test_sorted_merch_target_error_on_dimension_mismatch2(self):

        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [2.0, 2.0, 2.0, 2.0]}
            )
        )

        # note 5 values here, and 4 in inventory
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10]})
        )

        with self.assertRaises(ValueError):
            rule_target.sorted_merch_target(
                carbon_target=55,
                disturbance_production=mock_disturbance_production,
                inventory=mock_inventory,
                sort_value=series.from_pandas(
                    pd.Series([4, 3, 2, 1, 15])
                ),  # extra here
                efficiency=1.0,
                eligible=series.from_pandas(
                    pd.Series([True, True, True, True])
                ),
            )

    def test_sorted_merch_target_expected_result_with_efficiency(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {"age": [0, 20, 10, 30], "area": [1.0, 2.0, 1.0, 1.0]}
            )
        )
        mock_disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [10, 10, 10, 10]})
        )
        # with efficiency < 1.0 the disturbance production is lowered,
        # and all records will be split

        result = rule_target.sorted_merch_target(
            carbon_target=33,
            disturbance_production=mock_disturbance_production,
            inventory=mock_inventory,
            sort_value=series.from_pandas(pd.Series([4, 3, 2, 1])),
            efficiency=0.8,
            eligible=series.from_pandas(pd.Series([True, True, True, True])),
        )
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 1, 2, 3]
        )

        # efficiency*production causes this
        self.assertTrue(result.target["target_var"].to_list() == [8, 16, 8, 8])

        self.assertTrue(result.target["sort_var"].to_list() == [4, 3, 2, 1])

        # (0.8 * 10 + 0.8 * 20 + 0.8 * 10) == 32
        # (10 * x) == 33 - 32 == 1
        # x = 1/10

        # carbon_target = 0.8 * 3 * 10 + 1/10 = 25
        self.assertTrue(
            np.allclose(
                result.target["area_proportions"].to_numpy(),
                [0.8, 0.8, 0.8, 1 / 10],
            )
        )
        self.assertTrue(result.statistics["total_eligible_value"] == 40)
        self.assertTrue(result.statistics["total_achieved"] == 33)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 4)
        self.assertTrue(result.statistics["num_splits"] == 1)
        self.assertTrue(result.statistics["num_eligible"] == 4)

    def test_proportion_area_target(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame({"area": [60.0, 60.0, 60.0, 60.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, False]))
        result = rule_target.proportion_area_target(
            area_target_value=100, inventory=mock_inventory, eligible=eligible
        )
        self.assertTrue(
            result.target["target_var"].to_list() == [5 / 6 * 60, 5 / 6 * 60]
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(result.target["disturbed_index"].to_list() == [0, 2])
        self.assertTrue(
            result.target["area_proportions"].to_list() == [5 / 6, 5 / 6]
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == mock_inventory["area"].filter(eligible).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 100)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(
            result.statistics["num_records_disturbed"] == eligible.sum()
        )
        self.assertTrue(result.statistics["num_splits"] == eligible.sum())
        self.assertTrue(result.statistics["num_eligible"] == eligible.sum())

    def test_proportion_area_target_with_shortfall(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    # note 3rd value is 0 area
                    "area": [60.0, 60.0, 0.0, 60.0]
                }
            )
        )
        eligible = series.from_pandas(pd.Series([True, False, True, False]))
        result = rule_target.proportion_area_target(
            area_target_value=1000, inventory=mock_inventory, eligible=eligible
        )
        self.assertTrue(result.target["target_var"].to_list() == [60, 0])
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(result.target["disturbed_index"].to_list() == [0, 2])
        self.assertTrue(
            result.target["area_proportions"].to_list() == [1.0, 1.0]
        )
        self.assertTrue(
            result.statistics["total_eligible_value"]
            == mock_inventory["area"].filter(eligible).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 60)
        self.assertTrue(result.statistics["shortfall"] == 1000 - 60)
        self.assertTrue(
            result.statistics["num_records_disturbed"] == eligible.sum()
        )
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == eligible.sum())

    def test_proportion_area_target_with_all_zero_areas(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame({"area": [0.0, 0.0, 0.0, 0.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, False]))
        result = rule_target.proportion_area_target(
            area_target_value=1000, inventory=mock_inventory, eligible=eligible
        )
        self.assertTrue(result.target is None)
        self.assertTrue(result.statistics["total_eligible_value"] == 0.0)
        self.assertTrue(result.statistics["total_achieved"] == 0)
        self.assertTrue(result.statistics["shortfall"] == 1000)
        self.assertTrue(result.statistics["num_records_disturbed"] == 0)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == eligible.sum())

    def test_proportion_merch_target_eff_lt_1(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [12.0, 11.0, 10.0, 7.0],
                }
            )
        )
        carbon_target = 100.0
        disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [5.0, 5.0, 4.0, 10.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, True]))
        efficiency = 0.9
        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=efficiency,
            eligible=eligible,
        )

        # the sequence of eligible disturbance production is:
        # (area*production*efficiency)
        # 12.0 * 5.0 * 0.9 + 10.0 * 4.0 * 0.9 + 7.0 * 10.0 * 0.9
        # sum of the above is 54 + 36 + 63 = 153 (tC eligible production)
        # area proportion = 100/153 = 0.65359477

        self.assertTrue(
            result.target["target_var"].to_list()
            == [
                12.0 * 5.0 * 0.9 * 100 / 153,
                10.0 * 4.0 * 0.9 * 100 / 153,
                7.0 * 10.0 * 0.9 * 100 / 153,
            ]
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 2, 3]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list()
            == [100 / 153 * 0.9, 100 / 153 * 0.9, 100 / 153 * 0.9]
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == (
                mock_inventory["area"].filter(eligible)
                * disturbance_production["Total"].filter(eligible)
                * efficiency
            ).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 100)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 3)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_proportion_merch_target(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [12.0, 11.0, 10.0, 7.0],
                }
            )
        )
        carbon_target = 100.0
        disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [5.0, 5.0, 4.0, 10.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, True]))

        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=1.0,
            eligible=eligible,
        )

        total_eligible_production = (
            disturbance_production["Total"].filter(eligible)
            * mock_inventory["area"].filter(eligible)
        ).sum()
        target_proportion = carbon_target / total_eligible_production
        self.assertTrue(
            result.target["target_var"].to_list()
            == (
                disturbance_production["Total"].filter(eligible)
                * mock_inventory["area"].filter(eligible)
                * target_proportion
            ).to_list()
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 2, 3]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list()
            == [target_proportion] * eligible.sum()
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == (
                mock_inventory["area"].filter(eligible)
                * disturbance_production["Total"].filter(eligible)
            ).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == 100)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 3)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_proportion_merch_target_some_zero_production(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [12.0, 11.0, 10.0, 7.0],
                }
            )
        )
        carbon_target = 50.0
        disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [5.0, 5.0, 0.0, 10.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, True]))

        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=1.0,
            eligible=eligible,
        )

        total_eligible_production = (
            disturbance_production["Total"].filter(eligible)
            * mock_inventory["area"].filter(eligible)
        ).sum()
        target_proportion = carbon_target / total_eligible_production
        self.assertTrue(
            result.target["target_var"].to_list()
            == (
                disturbance_production["Total"].filter(eligible)
                * mock_inventory["area"].filter(eligible)
                * target_proportion
            ).to_list()
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 2, 3]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list()
            == [target_proportion] * eligible.sum()
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == (
                mock_inventory["area"].filter(eligible)
                * disturbance_production["Total"].filter(eligible)
            ).sum()
        )
        self.assertTrue(result.statistics["total_achieved"] == carbon_target)
        self.assertTrue(result.statistics["shortfall"] == 0)
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 3)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_proportion_merch_target_all_zero_production(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [12.0, 11.0, 10.0, 7.0],
                }
            )
        )
        carbon_target = 50.0
        # no disturbance production at all to meet the above target
        disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [0.0, 0.0, 0.0, 0.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, True]))

        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=1.0,
            eligible=eligible,
        )

        self.assertTrue(result.target is None)
        self.assertTrue(result.statistics["total_eligible_value"] == 0)
        self.assertTrue(result.statistics["total_achieved"] == 0)
        self.assertTrue(result.statistics["shortfall"] == carbon_target)
        self.assertTrue(result.statistics["num_records_disturbed"] == 0)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_proportion_merch_target_shortfall(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame(
                {
                    "area": [12.0, 11.0, 10.0, 7.0],
                }
            )
        )
        carbon_target = 500.0
        disturbance_production = dataframe.from_pandas(
            pd.DataFrame({"Total": [5.0, 5.0, 4.0, 10.0]})
        )
        eligible = series.from_pandas(pd.Series([True, False, True, True]))
        efficiency = 0.9
        result = rule_target.proportion_merch_target(
            carbon_target=carbon_target,
            disturbance_production=disturbance_production,
            inventory=mock_inventory,
            efficiency=efficiency,
            eligible=eligible,
        )

        # the sequence of eligible disturbance production is:
        # (area*production*efficiency)
        # 12.0 * 5.0 * 0.9 + 10.0 * 4.0 * 0.9 + 7.0 * 10.0 * 0.9
        # sum of the above is 54 + 36 + 63 = 153 (tC eligible production)
        # area proportion = 500/153 (>1.0)

        self.assertTrue(result.target["target_var"].sum())
        self.assertTrue(
            result.target["target_var"].to_list()
            == (
                mock_inventory["area"].filter(eligible)
                * disturbance_production["Total"].filter(eligible)
                * efficiency
            ).to_list()
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 2, 3]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list()
            == list(pd.Series([1.0, 1.0, 1.0]) * 0.9)
        )

        self.assertTrue(
            result.statistics["total_eligible_value"]
            == (
                mock_inventory["area"].filter(eligible)
                * disturbance_production["Total"].filter(eligible)
                * efficiency
            ).sum()
        )
        self.assertTrue(
            result.statistics["total_achieved"]
            == result.statistics["total_eligible_value"]
        )
        self.assertTrue(
            result.statistics["shortfall"]
            == (carbon_target - result.statistics["total_eligible_value"])
        )
        self.assertTrue(result.statistics["num_records_disturbed"] == 3)
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == 3)

    def test_proportion_sort_proportion_target(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame({"area": [10, 20, 30, 40, 50, 60]})
        )
        eligible = series.from_pandas(
            pd.Series([True, False, True, True, False, True])
        )
        proportion_target = 3 / 4

        result = rule_target.proportion_sort_proportion_target(
            proportion_target=proportion_target,
            inventory=mock_inventory,
            eligible=eligible,
        )

        self.assertTrue(
            result.target["target_var"].to_list()
            == [3 / 4, 3 / 4, 3 / 4, 3 / 4]
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [0, 2, 3, 5]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list()
            == [3 / 4, 3 / 4, 3 / 4, 3 / 4]
        )

        self.assertTrue(result.statistics["total_eligible_value"] is None)
        self.assertTrue(result.statistics["total_achieved"] is None)
        self.assertTrue(result.statistics["shortfall"] is None)
        self.assertTrue(result.statistics["num_records_disturbed"] == 4)
        self.assertTrue(result.statistics["num_splits"] == 4)
        self.assertTrue(result.statistics["num_eligible"] == 4)

    def test_proportion_sort_proportion_target_error_on_invalid_prop(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame({"area": [10, 20, 30, 40, 50, 60]})
        )
        eligible = series.from_pandas(
            pd.Series([True, False, True, True, False, True])
        )
        with self.assertRaises(ValueError):
            rule_target.proportion_sort_proportion_target(
                proportion_target=-0.001,
                inventory=mock_inventory,
                eligible=eligible,
            )
        with self.assertRaises(ValueError):
            rule_target.proportion_sort_proportion_target(
                proportion_target=1.001,
                inventory=mock_inventory,
                eligible=eligible,
            )

    def test_proportion_sort_proportion_target_w_prop_eq_1(self):
        mock_inventory = dataframe.from_pandas(
            pd.DataFrame({"area": [10, 20, 30, 40, 50, 60]})
        )
        eligible = series.from_pandas(
            pd.Series([False, False, True, True, False, True])
        )
        proportion_target = 1.0

        result = rule_target.proportion_sort_proportion_target(
            proportion_target=proportion_target,
            inventory=mock_inventory,
            eligible=eligible,
        )
        n_eligble = eligible.sum()
        self.assertTrue(
            result.target["target_var"].to_list() == [1.0] * n_eligble
        )
        self.assertTrue(pd.isnull(result.target["sort_var"].to_list()).all())
        self.assertTrue(
            result.target["disturbed_index"].to_list() == [2, 3, 5]
        )
        self.assertTrue(
            result.target["area_proportions"].to_list() == [1.0] * n_eligble
        )

        self.assertTrue(result.statistics["total_eligible_value"] is None)
        self.assertTrue(result.statistics["total_achieved"] is None)
        self.assertTrue(result.statistics["shortfall"] is None)
        self.assertTrue(
            result.statistics["num_records_disturbed"] == n_eligble
        )
        #  proportion=1 results in zero splits
        self.assertTrue(result.statistics["num_splits"] == 0)
        self.assertTrue(result.statistics["num_eligible"] == n_eligble)
