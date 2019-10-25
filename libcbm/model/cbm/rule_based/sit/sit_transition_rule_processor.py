from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based import rule_filter

def sit_transition_rule_iterator(sit_transitions, classifier_names):
    """Groups transition rules by classifiers, and eligibility criteria and
    yields the sequence of group_key, group.

    Args:
        sit_transitions (pandas.DataFrame): parsed sit_transitions. See
            :py:mod:`libcbm.input.sit.sit_transition_rule_parser`
        classifier_names (list): the list of classifier names which must
            correspond to the first len(classifier_names) columns of
            sit_transitions
    Raises:
        ValueError: the sum of the percent field for any grouped set of
            transition rule rows exceeded 100%
    """
    group_cols = classifier_names + \
        ["min_age", "max_age", "disturbance_type"]

    # group transition rules by their filter criteria
    # (classifier set, age range, disturbance type)
    grouping = sit_transitions.group_by(group_cols)
    for group_key, group in dict(iter(grouping)):
        if group.percent.sum() > 100:
            raise ValueError(
                "Greater than 100 percent sum for percent field in "
                f"grouped transition rules with: {group_key}")
        yield group_key, group


class SITTransitionRuleProcessor(object):

    def __init__(self, classifier_filter_builder):
        self.classifier_filter_builder = classifier_filter_builder

    def apply_transition_rule(self, tr_group_key, tr_group, transitioned,
                              classifiers, age, disturbance_type):

        state_filter_expression, state_filter_cols = \
            sit_stand_filter.create_state_filter_expression(
                tr_group_key, True)

        tr_filter = rule_filter.merge_filters(
            rule_filter.create_filter(
                expression=state_filter_expression,
                data={"age": age},
                columns=state_filter_cols),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    tr_group_key, classifiers.columns.tolist()),
                classifiers))



    # steps to process a grouped transition rule:
    # 1. filter by classifierset, age range, and disturbance type
    # 2. check for collisions with other transition rule groups
    #    (and raise an error if this occurs)
    # 3. create a single new record with the post transition rule
    #    cset, regen delay and age for each member of grouped transition rule.
    #


