"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import pandas as pd
from libcbm.model.cbm.rule_based import rule_filter


def create_split_proportions(tr_group_key, tr_group, group_error_max):
    # dealing with a couple of cases here:
    #
    # if the sum of the percent column in the specified group is less then
    # 100% then the number of splits is len(tr_group)+1 since the remainder
    # is allowed and is modelled as "unchanged" as far as transitioning
    # classifiers, etc.

    percent_sum = tr_group.percent.sum()
    if abs(percent_sum - 100) < group_error_max:
        return tr_group.percent / percent_sum
    elif percent_sum > 100:
        raise ValueError(
            f"total percent ({percent_sum}) in transition rule group "
            f"{tr_group_key} exceeds 100%")
    else:
        remainder = 100 - percent_sum
        appended_percent = tr_group.percent.append(pd.Series([remainder]))
        return list(appended_percent / appended_percent.sum())


class TransitionRuleProcessor(object):

    def __init__(self, classifier_filter_builder, state_variable_filter_func,
                 classifiers_config, grouped_percent_err_max, wildcard,
                 transition_classifier_postfix):
        self.wildcard = wildcard
        self.transition_classifier_postfix = transition_classifier_postfix
        self.state_variable_filter_func = state_variable_filter_func
        self.classifier_filter_builder = classifier_filter_builder
        self.grouped_percent_err_max = grouped_percent_err_max
        self.classifiers_config = classifiers_config
        self.classifier_names = [
            x["name"] for x in self.classifiers_config["classifiers"]]
        self.classifier_value_lookup = {
            x["name"]: self._get_classifier_value_index(x["id"])
            for x in self.classifiers_config["classifiers"]}

    def _get_classifier_value_index(self, classifier_id):
        return {
            x["value"]: x["id"] for x
            in self.classifiers_config["classifier_values"]
            if x["classifier_id"] == classifier_id}

    def _filter_stands(self, tr_group_key, cbm_vars):

        dist_type_target = tr_group_key["disturbance_type_id"]
        classifier_set = [
            tr_group_key[x]
            for x in cbm_vars.classifiers.columns.tolist()]
        tr_filter = rule_filter.merge_filters(
            self.state_variable_filter_func(tr_group_key, cbm_vars.state),
            self.classifier_filter_builder.create_classifiers_filter(
                classifier_set,
                cbm_vars.classifiers),
            rule_filter.create_filter(
                expression=f"(disturbance_type_id == {dist_type_target})",
                data={"disturbance_type_id": cbm_vars.params.disturbance_type},
                columns=["disturbance_type_id"]))

        filter_result = rule_filter.evaluate_filter(tr_filter)
        return filter_result

    def _get_transition_classifier_set(self, transition_rule):
        for classifier_name in self.classifier_names:
            transition_classifier_value = transition_rule[
                classifier_name + self.transition_classifier_postfix]
            if transition_classifier_value == self.wildcard:
                continue
            transition_id = self.classifier_value_lookup[
                classifier_name][transition_classifier_value]
            yield classifier_name, transition_id

    def apply_transition_rule(self, tr_group_key, tr_group, transition_mask,
                              cbm_vars):
        """Apply the specified transition rule group to the simulation
        variables, updating classifier values, and returning the transition
        rule variables reset age, and regeneration delay.  For each member of
        the transition rule group a split of the simulation variables will
        occur with area being reduced according to the "percent" column in
        the member transition rules.

        Args:
            tr_group_key (dict): the common key for the grouped transition
                rules.
            tr_group (pandas.DataFrame): the grouped transition rules, where
                each row is a member.
            transition_mask (numpy.ndarray): a boolean mask indicating when
                true that the correspoding index has already been transitioned.
                This is used to detect transition rule criteria collisions.
            cbm_vars (object): CBM simulation variables and state

        Raises:
            ValueError: a transition rule criteria resulted in the selection of
                stands targetted by at least one other transition rule

        Returns:
            tuple:

                - transition_mask: the specified transition_mask parameter is
                    returned altered with the indices transitioned by this
                    function call.
                - cbm_vars: updated and potentially expanded cbm variables and
                    state

        """
        filter_result = self._filter_stands(tr_group_key, cbm_vars)

        if np.logical_and(transition_mask, filter_result).any():
            # this indicates that a transition rule has collided with another
            # transition rule, which is possible when overlapping criteria are
            # specified (wildcards, age ranges etc.)  This is a simplistic,
            # but safe solution for this possible issue. Another approach might
            # be to prioritize overlapping groups instead.
            raise ValueError(
                "overlapping transition rule criteria detected: "
                f"{tr_group_key}")

        # sets the transitioned array with the transition filter result
        transition_mask = np.logical_or(transition_mask, filter_result)
        transition_mask_output = transition_mask.copy()

        proportions = create_split_proportions(
            tr_group_key, tr_group, self.grouped_percent_err_max)

        # storage for split records
        classifier_split = pd.DataFrame()
        inventory_split = pd.DataFrame()
        pool_split = pd.DataFrame()
        state_split = pd.DataFrame()
        params_split = pd.DataFrame()
        flux_split = pd.DataFrame()

        for i_proportion, proportion in enumerate(proportions):
            if i_proportion == 0:
                continue
                # the first proportion corresponds to teh source data
                # for all remaining splits, so the must be updated last so
                # those changes don't propagate into the splits.

            # For all proportions other than the first we need to make
            # a copy of each of the state variables to split off the
            # percentage for the current group member
            pools = cbm_vars.pools[filter_result].copy()
            flux = cbm_vars.flux_indicators[filter_result].copy()
            params = cbm_vars.params[filter_result].copy()
            state = cbm_vars.state[filter_result].copy()
            inventory = cbm_vars.inventory[filter_result].copy()
            classifiers = cbm_vars.classifiers[filter_result].copy()
            transition_mask_output = np.concatenate([
                transition_mask_output,
                transition_mask[filter_result].copy()])

            # set the area for the split portion according to the current
            # group member proportion
            inventory.area = inventory.area * proportion

            # if the current proportion is the remainder of 100 minus the
            # group's percentage sums, then this split portion will not be
            # transitioned, meaning the classifier set is not changed, and
            # the reset age, regeneration delay parameters will not take
            # effect for this split portion
            is_transition_split = i_proportion < tr_group.shape[0]
            if is_transition_split:

                transition_classifier_ids = \
                    self._get_transition_classifier_set(
                        transition_rule=tr_group.iloc[i_proportion])
                # update the split classifiers with the transitioned value
                for classifier_name, value_id in transition_classifier_ids:
                    classifiers[classifier_name] = value_id

                state.regeneration_delay = \
                    tr_group.iloc[i_proportion].regeneration_delay

                params.reset_age = tr_group.iloc[i_proportion].reset_age

            classifier_split = classifier_split.append(classifiers)
            inventory_split = inventory_split.append(inventory)
            pool_split = pool_split.append(pools)
            state_split = state_split.append(state)
            params_split = params_split.append(params)
            flux_split = flux_split.append(flux)

        # for the first index use the existing matched records
        transition_classifier_ids = \
            self._get_transition_classifier_set(
                transition_rule=tr_group.iloc[0])
        for classifier_name, value_id in transition_classifier_ids:
            cbm_vars.classifiers.loc[
                filter_result, classifier_name] = value_id
        if proportions[0] < 1.0:
            cbm_vars.inventory.loc[filter_result, "area"] = \
                cbm_vars.inventory.loc[filter_result, "area"] * \
                proportions[0]

        cbm_vars.state.loc[filter_result, "regeneration_delay"] = \
            tr_group.iloc[0].regeneration_delay

        cbm_vars.params.loc[filter_result, "reset_age"] = \
            tr_group.iloc[0].reset_age

        if len(proportions) > 1:
            cbm_vars.classifiers = cbm_vars.classifiers.append(
                classifier_split).reset_index(drop=True)
            cbm_vars.inventory = cbm_vars.inventory.append(
                inventory_split).reset_index(drop=True)
            cbm_vars.pools = cbm_vars.pools.append(
                pool_split).reset_index(drop=True)
            cbm_vars.state = cbm_vars.state.append(
                state_split).reset_index(drop=True)
            cbm_vars.params = cbm_vars.params.append(
                params_split).reset_index(drop=True)
            cbm_vars.flux_indicators = cbm_vars.flux_indicators.append(
                flux_split).reset_index(drop=True)

        return transition_mask_output, cbm_vars
