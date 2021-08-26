# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd
from libcbm.model.cbm.rule_based.transition_rule_processor import \
    TransitionRuleProcessor
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor
from libcbm.model.cbm.rule_based.sit import sit_event_processor


def sit_rule_based_processor_factory(cbm, random_func, classifiers_config,
                                     classifier_aggregates, sit_events,
                                     sit_transitions, tr_constants,
                                     sit_disturbance_eligibilities):

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
        event_processor, tr_processor, sit_events, sit_transitions,
        sit_disturbance_eligibilities)


class SITRuleBasedProcessor():

    def __init__(self, event_processor, transition_rule_processor,
                 sit_events, sit_transitions, sit_eligibilities):
        self.event_processor = event_processor
        self.transition_rule_processor = transition_rule_processor
        self.sit_events = sit_events
        self.sit_eligibilities = sit_eligibilities
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
            cbm_vars=cbm_vars,
            sit_eligibilities=self.sit_eligibilities)
        self.sit_event_stats_by_timestep[time_step] = stats_df
        return cbm_vars

    def pre_dynamic_func(self, time_step, cbm_vars):
        cbm_vars = self.dist_func(time_step, cbm_vars)
        cbm_vars = self.tr_func(cbm_vars)
        cbm_vars.classifiers = pd.DataFrame(
            columns=cbm_vars.classifiers.columns,
            data=np.ascontiguousarray(
                cbm_vars.classifiers.to_numpy(dtype="int32")))
        return cbm_vars
