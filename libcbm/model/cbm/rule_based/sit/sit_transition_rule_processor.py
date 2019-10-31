import numpy as np
from libcbm.input.sit import sit_transition_rule_parser
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based import rule_filter


def get_pre_dynamics_func(sit_transition_processor, sit_transitions):
    """Gets a function for applying SIT transition rules in a CBM
    timestep loop.

        The returned function can be used as the
        pre_dynamics_func argument of
        :py:func:`libcbm.model.cbm.cbm_simulator.simulate’`

    Args:
        sit_transition_processor (SITTransitionRuleProcessor):
            instance of an object to apply sit transitions to the simulation
            state.
        sit_transitions (pandas.DataFrame): table of SIT formatted transitions.
            Expected format is the same as the return value of:
            :py:func:`libcbm.input.sit.sit_transition_rule_parser.parse`

    Returns:
        func: a function of 2 parameters:

            1. time_step: the simulation time step which is used to select
                sit_events by the time_step column.
            2. cbm_vars: an object containing CBM simulation variables and
                parameters.  Formatted the same as the return value of
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`

            The function's return value is a copy of the input cbm_vars
            with changes applied according to the sit_events for the
            specified timestep.
    """
    def sit_transition_pre_dynamics_func(_, cbm_vars):
        cbm_vars = sit_transition_processor.process_transition_rules(
            sit_transitions, cbm_vars)
        return cbm_vars
    return sit_transition_pre_dynamics_func


def state_variable_filter_func(tr_group_key, state_variables):
    """Create a filter based on transition rule state criteria for setting
    stands eligible or ineligible for transition.

    Args:
        tr_group_key (dict): dictionary of values common to a transition rule
            group
        state_variables (pandas.DataFrame): table of state values for the
            current simulation for which to create a filter.

    Returns:
        object: a filter object
    """
    state_filter_expression, state_filter_cols = \
        sit_stand_filter.create_state_filter_expression(
            tr_group_key, True)
    return rule_filter.create_filter(
        expression=state_filter_expression,
        data={"age": state_variables.age},
        columns=state_filter_cols)


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
        ["min_age", "max_age", "disturbance_type_id"]

    # group transition rules by their filter criteria
    # (classifier set, age range, disturbance type)
    grouping = sit_transitions.groupby(group_cols)
    group_error_max = sit_transition_rule_parser.GROUPED_PERCENT_ERR_MAX
    for group_key, group in dict(list(grouping)).items():
        group_key_dict = dict(zip(group_cols, group_key))
        if group.percent.sum() > 100 + group_error_max:
            raise ValueError(
                "Greater than 100 percent sum for percent field in "
                f"grouped transition rules with: {group_key_dict}")
        yield group_key_dict, group


class SITTransitionRuleProcessor:

    def __init__(self, transition_rule_processor):
        self.transition_rule_processor = transition_rule_processor

    def process_transition_rules(self, sit_transitions, cbm_vars):

        classifiers = cbm_vars.classifiers
        n_stands = classifiers.shape[0]
        classifier_names = classifiers.columns.tolist()
        transition_iterator = sit_transition_rule_iterator(
            sit_transitions, classifier_names)
        transition_mask = np.zeros(n_stands, dtype=bool)
        for tr_group_key, tr_group in transition_iterator:
            transition_mask, cbm_vars = \
                self.transition_rule_processor.apply_transition_rule(
                    tr_group_key, tr_group, transition_mask, cbm_vars)
        return cbm_vars
