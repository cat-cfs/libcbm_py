# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from libcbm.model.cbm.rule_based.transition_rule_processor import \
    TransitionRuleProcessor
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor
from libcbm.model.cbm.rule_based.sit import sit_event_processor


def sit_rule_based_processor_factory(cbm, random_func, classifiers_config,
                                     classifier_aggregates, sit_events,
                                     sit_transitions, tr_constants):

    classifier_filter = ClassifierFilter(
        classifiers_config=classifiers_config,
        classifier_aggregates=classifier_aggregates)

    state_filter_func = \
        sit_transition_rule_processor.state_variable_filter_func

    tr_processor = sit_transition_rule_processor.SITTransitionRuleProcessor(
        TransitionRuleProcessor(
            classifier_filter_builder=classifier_filter,
            state_variable_filter_func=state_filter_func,
            classifiers_config=classifiers_config,
            grouped_percent_err_max=tr_constants.group_err_max,
            wildcard=tr_constants.wildcard,
            transition_classifier_postfix=tr_constants.classifier_value_postfix
        ))

    event_processor = sit_event_processor.SITEventProcessor(
        model_functions=cbm.model_functions,
        compute_functions=cbm.compute_functions,
        classifier_filter_builder=classifier_filter,
        random_generator=random_func)

    return SITRuleBasedProcessor(
        event_processor, tr_processor, sit_events, sit_transitions)


#def get_pre_dynamics_func(sit_event_processor, sit_events):
#    """Gets a function for applying SIT rule based events in a CBM
#    timestep loop.
#
#    The returned function can be used as the
#    pre_dynamics_func argument of
#    :py:func:`libcbm.model.cbm.cbm_simulator.simulate’`
#
#    Args:
#        sit_event_processor (SITEventProcessor): Instance of
#            :py:class:`SITEventProcessor` for computing sit rule based
#            disturbance events.
#        sit_events (pandas.DataFrame): table of SIT formatted events.
#            Expected format is the same as the return value of:
#            :py:func:`libcbm.input.sit.sit_disturbance_event_parser.parse`
#    Returns:
#        func: a function of 2 parameters:
#
#            1. time_step: the simulation time step which is used to select
#                sit_events by the time_step column.
#            2. cbm_vars: an object containing CBM simulation variables and
#                parameters.  Formatted the same as the return value of
#                :py:func:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
#
#            The function's return value is a copy of the input cbm_vars
#            with changes applied according to the sit_events for the
#            specified timestep.
#
#    """
#    def sit_events_pre_dynamics_func(time_step, cbm_vars, stats_func):
#        cbm_vars, stats_df = sit_event_processor.process_events(
#            time_step=time_step,
#            sit_events=sit_events,
#            cbm_vars=cbm_vars)
#        stats_func(time_step, stats_df)
#        return cbm_vars
#
#    return sit_events_pre_dynamics_func

#def get_pre_dynamics_func(sit_transition_processor, sit_transitions):
#    """Gets a function for applying SIT transition rules in a CBM
#    timestep loop.
#
#        The returned function can be used as the
#        pre_dynamics_func argument of
#        :py:func:`libcbm.model.cbm.cbm_simulator.simulate’`
#
#    Args:
#        sit_transition_processor (SITTransitionRuleProcessor):
#            instance of an object to apply sit transitions to the simulation
#            state.
#        sit_transitions (pandas.DataFrame): table of SIT formatted transitions.
#            Expected format is the same as the return value of:
#            :py:func:`libcbm.input.sit.sit_transition_rule_parser.parse`
#
#    Returns:
#        func: a function of 2 parameters:
#
#            1. time_step: the simulation time step which is used to select
#                sit_events by the time_step column.
#            2. cbm_vars: an object containing CBM simulation variables and
#                parameters.  Formatted the same as the return value of
#                :py:func:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
#
#            The function's return value is a copy of the input cbm_vars
#            with changes applied according to the sit_events for the
#            specified timestep.
#    """
#    def sit_transition_pre_dynamics_func(_, cbm_vars):
#        cbm_vars = sit_transition_processor.process_transition_rules(
#            sit_transitions, cbm_vars)
#        return cbm_vars
#    return sit_transition_pre_dynamics_func

class SITRuleBasedProcessor():

    def __init__(self, event_processor, transition_rule_processor,
                 sit_events, sit_transitions):
        self.event_processor = event_processor
        self.transition_rule_processor = transition_rule_processor
        self.sit_events = sit_events
        self.sit_event_stats_by_timestep = {}
        self.sit_transitions = sit_transitions

    def tr_func(self, cbm_vars):
        cbm_vars = self.transition_rule_processor.process_transition_rules(
            self.sit_transitions, cbm_vars)
        return cbm_vars

    def dist_func(self, time_step, cbm_vars):
        cbm_vars, stats_df = self.event_processor.process_events(
            time_step=time_step,
            sit_events=self.sit_events,
            cbm_vars=cbm_vars)
        self.sit_event_stats_by_timestep[time_step] = stats_df
        return cbm_vars

    def pre_dynamic_func(self, time_step, cbm_vars):
        cbm_vars = self.dist_func(time_step, cbm_vars)
        cbm_vars = self.tr_func(cbm_vars)
        return cbm_vars
