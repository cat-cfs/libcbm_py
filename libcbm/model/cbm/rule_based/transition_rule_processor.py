import numpy as np
import pandas as pd
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
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
        return appended_percent / appended_percent.sum()


class TransitionRuleProcessor(object):

    def __init__(self, classifier_filter_builder, classifiers_config,
                 grouped_percent_err_max):
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

    def filter_stands(self, tr_group_key, inventory, disturbance_type):
        state_filter_expression, state_filter_cols = \
            sit_stand_filter.create_state_filter_expression(
                tr_group_key, True)

        classifiers = inventory.classifiers
        disturbance_type_target = tr_group_key["disturbance_type"]
        tr_filter = rule_filter.merge_filters(
            rule_filter.create_filter(
                expression=state_filter_expression,
                data={"age": inventory.age},
                columns=state_filter_cols),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    tr_group_key, classifiers.columns.tolist()),
                classifiers),
            rule_filter.create_filter(
                expression=f"(disturbance_type == {disturbance_type_target})",
                data={"disturbance_type": disturbance_type},
                columns=["disturbance_type"]))

        filter_result = rule_filter.evaluate_filter(tr_filter)
        return filter_result

    def apply_transition_rule(self, tr_group_key, tr_group, transitioned,
                              disturbance_type, classifiers, inventory, pools,
                              state_variables):

        filter_result = self.filter_stands(
            tr_group_key, inventory, disturbance_type)

        if np.logical_and(transitioned, filter_result).any():
            # this indicates that a transition rule has collided with another
            # transition rule, which is possible when overlapping criteria are
            # specified (wildcards, age ranges etc.)  This is a simplistic,
            # but safe solution for this possible issue. Another approach might
            # be to prioritize overlapping groups instead.
            raise ValueError(
                "overlapping transition rule criteria detected: "
                f"{tr_group_key}")

        # sets the transitioned array with the transition filter result
        transitioned = np.logical_or(transitioned, filter_result)

        proportions = create_split_proportions(
            tr_group_key, tr_group, self.grouped_percent_err_max)

        transitioned_classifier_rows = classifiers[filter_result].copy()
        transition_classifier_append_result = pd.DataFrame()
        for i_proportion, proportion in enumerate(proportions):
            if i_proportion == 0:
                # for the first index use the existing matched inventory record
                transition_clasifier_ids = self.get_transition_classifier_set(
                    transition_rule=tr_group.iloc[i_proportion])
                for k, v in transition_clasifier_ids:
                    classifiers.loc[filter_result, k] = v

            elif i_proportion == tr_group.shape[0]:
                # if a proportion was created for the 100-grouped percentage
                # remainder, then make a copy with no transition
                transition_classifier_append_result = \
                    transition_classifier_append_result.append(
                        transitioned_classifier_rows)
            else:
                transition_clasifier_ids = self.get_transition_classifier_set(
                    transition_rule=tr_group.iloc[i_proportion])
                transition_copy_rows = transitioned_classifier_rows.copy()
                for k, v in transition_clasifier_ids:
                    transition_copy_rows[k] = v

    def get_transition_classifier_set(self, transition_rule):
        result = {}
        for classifier_name in self.classifier_names:
            transition_classifier_value = transition_rule[
                classifier_name + "_tr"]
            if transition_classifier_value == "?":
                continue
            transition_id = self.classifier_value_lookup[
                classifier_name][transition_classifier_value]
            result[classifier_name] = transition_id
        return result
