from libcbm.input.sit import sit_transition_rule_parser


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
    group_error_max = sit_transition_rule_parser.GROUPED_PERCENT_ERR_MAX
    for group_key, group in dict(iter(grouping)):
        group_key_dict = dict(zip(group_cols, group_key))
        if group.percent.sum() > 100 + group_error_max:
            raise ValueError(
                "Greater than 100 percent sum for percent field in "
                f"grouped transition rules with: {group_key_dict}")
        yield group_key_dict, group

