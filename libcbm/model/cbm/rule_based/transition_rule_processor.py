# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Callable
from typing import Iterable
from typing import Tuple
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.rule_filter import RuleFilter
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter


def create_split_proportions(
    tr_group_key: dict, tr_group: DataFrame, group_error_max: float
) -> list[float]:
    """Create proportions

    Args:
        tr_group_key (dict): the composite key common to the transition
            rule group members
        tr_group (DataFrame): The table of transition rule group
            members
        group_error_max (float): used as a threshold to test if the group's
            total percentage exceeds 100 percent.

    Raises:
        ValueError:  Thrown if the absolute difference of total percentage
            minus 100 percent is greater than the group_error_max threshold.

    Returns:
        list: a list of proportions whose sum is 1.0 for splitting records
            for transition
    """
    # dealing with a couple of cases here:
    #
    # if the sum of the percent column in the specified group is less then
    # 100% then the number of splits is len(tr_group)+1 since the remainder
    # is allowed and is modelled as "unchanged" as far as transitioning
    # classifiers, etc.

    percent_sum = tr_group.percent.sum()
    if abs(percent_sum - 100) < group_error_max:
        return list(tr_group.percent / percent_sum)
    elif percent_sum > 100:
        raise ValueError(
            f"total percent ({percent_sum}) in transition rule group "
            f"{tr_group_key} exceeds 100%"
        )
    else:
        remainder = 100 - percent_sum
        appended_percent = tr_group.percent.append(Series([remainder]))
        return list(appended_percent / appended_percent.sum())


class TransitionRuleProcessor(object):
    def __init__(
        self,
        classifier_filter_builder: ClassifierFilter,
        state_variable_filter_func: Callable[[dict, DataFrame], RuleFilter],
        classifiers_config: dict[str, list],
        grouped_percent_err_max: float,
        wildcard: str,
        transition_classifier_postfix: str,
    ):
        self.wildcard = wildcard
        self.transition_classifier_postfix = transition_classifier_postfix
        self.state_variable_filter_func = state_variable_filter_func
        self.classifier_filter_builder = classifier_filter_builder
        self.grouped_percent_err_max = grouped_percent_err_max
        self.classifiers_config = classifiers_config
        self.classifier_names = [
            x["name"] for x in self.classifiers_config["classifiers"]
        ]
        self.classifier_value_lookup = {
            x["name"]: self._get_classifier_value_index(x["id"])
            for x in self.classifiers_config["classifiers"]
        }

    def _get_classifier_value_index(self, classifier_id: int) -> dict:
        return {
            x["value"]: x["id"]
            for x in self.classifiers_config["classifier_values"]
            if x["classifier_id"] == classifier_id
        }

    def _filter_stands(
        self, tr_group_key: dict, cbm_vars: CBMVariables
    ) -> Series:

        dist_type_target = tr_group_key["disturbance_type_id"]
        classifier_set = [
            tr_group_key[x] for x in cbm_vars.classifiers.columns
        ]
        tr_filters = [
            self.state_variable_filter_func(tr_group_key, cbm_vars.state),
            self.classifier_filter_builder.create_classifiers_filter(
                classifier_set, cbm_vars.classifiers
            ),
            rule_filter.create_filter(
                expression=f"(disturbance_type_id == {dist_type_target})",
                data={
                    "disturbance_type_id": cbm_vars.parameters.disturbance_type
                },
            ),
        ]

        filter_result = rule_filter.evaluate_filters(*tr_filters)
        return filter_result

    def _get_transition_classifier_set(
        self, transition_rule: dict
    ) -> Iterable[Tuple[str, int]]:
        for classifier_name in self.classifier_names:
            transition_classifier_value = transition_rule[
                classifier_name + self.transition_classifier_postfix
            ]
            if transition_classifier_value == self.wildcard:
                continue
            transition_id = self.classifier_value_lookup[classifier_name][
                transition_classifier_value
            ]
            yield classifier_name, transition_id

    def apply_transition_rule(
        self,
        tr_group_key: dict,
        tr_group: DataFrame,
        transition_mask: Series,
        cbm_vars: CBMVariables,
    ) -> Tuple[Series, CBMVariables]:
        """Apply the specified transition rule group to the simulation
        variables, updating classifier values, and returning the transition
        rule variables reset age, and regeneration delay.  For each member of
        the transition rule group a split of the simulation variables will
        occur with area being reduced according to the "percent" column in
        the member transition rules.

        Args:
            tr_group_key (dict): the common key for the grouped transition
                rules.
            tr_group (DataFrame): the grouped transition rules, where
                each row is a member.
            transition_mask (Series): a boolean mask indicating when
                true that the correspoding index has already been transitioned.
                This is used to detect transition rule criteria collisions.
            cbm_vars (CBMVariables): CBM simulation variables and state

        Raises:
            ValueError: a transition rule criteria resulted in the selection of
                stands targeted by at least one other transition rule

        Returns:
            tuple:

                - transition_mask: the specified transition_mask parameter is
                    returned altered with the indices transitioned by this
                    function call.
                - cbm_vars: updated and potentially expanded cbm variables and
                    state

        """
        filtered = self._filter_stands(tr_group_key, cbm_vars)

        # sets the transitioned array with the transition filter result
        eligible = dataframe.logical_and(
            dataframe.logical_not(transition_mask), filtered
        )

        transition_mask_output = dataframe.logical_or(
            transition_mask, eligible
        )

        if not eligible.any():
            return transition_mask_output, cbm_vars
        eligible_idx = dataframe.indices_nonzero(eligible)

        proportions = create_split_proportions(
            tr_group_key, tr_group, self.grouped_percent_err_max
        )

        # storage for split records
        classifier_split = None
        inventory_split = None
        pool_split = None
        state_split = None
        params_split = None
        flux_split = None

        for i_proportion, proportion in enumerate(proportions):
            if i_proportion == 0:
                continue
                # the first proportion corresponds to the source data
                # for all remaining splits, so it must be updated last so
                # those changes don't propagate into the splits.

            # For all proportions other than the first we need to make
            # a copy of each of the state variables to split off the
            # percentage for the current group member
            pools = cbm_vars.pools.filter(eligible)
            flux = cbm_vars.flux.filter(eligible)
            params = cbm_vars.parameters.filter(eligible)
            state = cbm_vars.state.filter(eligible)
            inventory = cbm_vars.inventory.filter(eligible)
            classifiers = cbm_vars.classifiers.filter(eligible)
            transition_mask_output = dataframe.concat_series(
                [
                    transition_mask_output,
                    dataframe.make_boolean_series(
                        init=True, size=pools.shape[0]
                    ),
                ]
            )

            # set the area for the split portion according to the current
            # group member proportion
            inventory["area"] = inventory["area"] * proportion

            # if the current proportion is the remainder of 100 minus the
            # group's percentage sums, then this split portion will not be
            # transitioned, meaning the classifier set is not changed, and
            # the reset age, regeneration delay parameters will not take
            # effect for this split portion
            is_transition_split = i_proportion < tr_group.n_rows
            if is_transition_split:

                transition_classifier_ids = (
                    self._get_transition_classifier_set(
                        transition_rule=tr_group.at(i_proportion)
                    )
                )
                # update the split classifiers with the transitioned value
                for classifier_name, value_id in transition_classifier_ids:
                    classifiers[classifier_name] = value_id

                state.assign(
                    "regeneration_delay",
                    int(tr_group["regeneration_delay"].at(i_proportion)),
                )

                params.assign(
                    "reset_age", int(tr_group["reset_age"].at(i_proportion))
                )

            classifier_split = dataframe.concat_data_frame(
                [classifier_split, classifiers]
            )
            inventory_split = dataframe.concat_data_frame(
                [inventory_split, inventory]
            )
            pool_split = dataframe.concat_data_frame([pool_split, pools])
            state_split = dataframe.concat_data_frame([state_split, state])
            params_split = dataframe.concat_data_frame([params_split, params])
            flux_split = dataframe.concat_data_frame([flux_split, flux])

        # for the first index in the tr_group use the existing matched records
        transition_classifier_ids = self._get_transition_classifier_set(
            transition_rule=tr_group.at(0)
        )

        for classifier_name, value_id in transition_classifier_ids:
            cbm_vars.classifiers.assign(
                classifier_name, value_id, eligible_idx
            )

        if proportions[0] < 1.0:

            cbm_vars.inventory.assign(
                "area",
                cbm_vars.inventory["area"].take(eligible_idx) * proportions[0],
                eligible_idx,
            )

        cbm_vars.state.assign(
            "regeneration_delay",
            int(tr_group["regeneration_delay"].at(0)),
            eligible_idx,
        )

        cbm_vars.parameters.assign(
            "reset_age", int(tr_group["reset_age"].at(0)), eligible_idx
        )

        if len(proportions) > 1:
            cbm_vars.classifiers = dataframe.concat_data_frame(
                [cbm_vars.classifiers, classifier_split]
            )
            cbm_vars.inventory = dataframe.concat_data_frame(
                [cbm_vars.inventory, inventory_split]
            )
            cbm_vars.pools = dataframe.concat_data_frame(
                [cbm_vars.pools, pool_split]
            )
            cbm_vars.state = dataframe.concat_data_frame(
                [cbm_vars.state, state_split]
            )
            cbm_vars.parameters = dataframe.concat_data_frame(
                [cbm_vars.parameters, params_split]
            )
            cbm_vars.flux = dataframe.concat_data_frame(
                [cbm_vars.flux, flux_split]
            )

        return transition_mask_output, cbm_vars
