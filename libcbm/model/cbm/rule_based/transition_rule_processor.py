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

    def __init__(self, classifier_filter_builder):
        self.classifier_filter_builder = classifier_filter_builder

    def apply_transition_rule(self, tr_group_key, tr_group, transitioned,
                              disturbance_type, classifiers, inventory, pools,
                              state_variables):

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

        if np.logical_and(transitioned, filter_result).any():
            # this indicates that a transition rule has collided with another
            # transition rule, which is possible when overlapping criteria are
            # specified (wildcards, age ranges etc.)  This is a simplistic,
            # but safe solution for this possible issue. Another approach might
            # be to prioritize overlapping groups instead.
            raise ValueError("overlapping transition rule criteria detected: "
                             f"{tr_group_key}")

        # sets the transitioned array with the transition filter result
        transitioned = np.logical_or(transitioned, filter_result)

        proportions = create_split_proportions(tr_group_key, tr_group)

