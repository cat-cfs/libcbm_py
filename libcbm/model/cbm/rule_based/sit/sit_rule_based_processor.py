# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable
from libcbm.model.cbm.cbm_model import CBM
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.storage.series import Series
from libcbm.model.cbm.rule_based.transition_rule_processor import (
    TransitionRuleProcessor,
)
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter

from libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor import (
    SITTransitionRuleProcessor,
)
from libcbm.model.cbm.rule_based.sit.sit_event_processor import (
    SITEventProcessor,
)


class TransitionRuleConstants:
    def __init__(
        self,
        group_error_max: float,
        wildcard: str,
        classifier_value_postfix: str,
    ):
        self._group_error_max = group_error_max
        self._wildcard = wildcard
        self._classifier_value_postfix = classifier_value_postfix

    @property
    def wildcard(self) -> str:
        """Gets the wildcard symbol ("?" in cbm3)"""
        return self._wildcard

    @property
    def group_error_max(self) -> float:
        """Gets the maximum deviation from 1.0 when taking the sum of
        transition rule proportions in a transition rule group.
        """
        return self._group_error_max

    @property
    def classifier_value_postfix(self) -> str:
        """
        Gets the postfix added to the transition-to classifier columns
        """
        return self._classifier_value_postfix


class SITRuleBasedProcessor:
    def __init__(
        self,
        event_processor: SITEventProcessor,
        transition_rule_processor: SITTransitionRuleProcessor,
        sit_events: pd.DataFrame,
        sit_transitions: pd.DataFrame,
        sit_eligibilities: pd.DataFrame,
        reset_parameters: bool,
    ):
        self.event_processor = event_processor
        self.transition_rule_processor = transition_rule_processor
        self.sit_events = sit_events
        self.sit_eligibilities = sit_eligibilities
        self.sit_event_stats_by_timestep = {}
        self.sit_transitions = sit_transitions
        self._reset_parameters = reset_parameters

    def tr_func(self, cbm_vars: CBMVariables) -> CBMVariables:
        cbm_vars = self.transition_rule_processor.process_transition_rules(
            self.sit_transitions, cbm_vars, self.sit_eligibilities
        )
        return cbm_vars

    def dist_func(
        self, time_step: int, cbm_vars: CBMVariables
    ) -> CBMVariables:
        cbm_vars, stats_df = self.event_processor.process_events(
            time_step=time_step,
            sit_events=self.sit_events,
            cbm_vars=cbm_vars,
            sit_eligibilities=self.sit_eligibilities,
        )
        self.sit_event_stats_by_timestep[time_step] = stats_df
        return cbm_vars

    def pre_dynamics_func(
        self, time_step: int, cbm_vars: CBMVariables
    ) -> CBMVariables:
        if self._reset_parameters:
            cbm_vars.parameters["disturbance_type"].assign(np.int32(0))
            cbm_vars.parameters["reset_age"].assign(np.int32(-1))
        cbm_vars = self.dist_func(time_step, cbm_vars)
        cbm_vars = self.tr_func(cbm_vars)
        return cbm_vars


def sit_rule_based_processor_factory(
    cbm: CBM,
    random_func: Callable[[int], Series],
    classifiers_config: dict[str, list],
    classifier_aggregates: pd.DataFrame,
    sit_events: pd.DataFrame,
    sit_transitions: pd.DataFrame,
    tr_constants: TransitionRuleConstants,
    sit_eligibilities: pd.DataFrame,
    reset_parameters: bool,
    disturbance_type_map: dict,
) -> SITRuleBasedProcessor:
    classifier_filter = ClassifierFilter(
        classifiers_config=classifiers_config,
        classifier_aggregates=classifier_aggregates,
    )

    tr_processor = SITTransitionRuleProcessor(
        TransitionRuleProcessor(
            classifiers_config=classifiers_config,
            wildcard=tr_constants.wildcard,
            transition_classifier_postfix=tr_constants.classifier_value_postfix,  # noqa 501
        ),
        classifier_filter=classifier_filter,
        group_error_max=tr_constants.group_error_max,
    )

    event_processor = SITEventProcessor(
        cbm=cbm,
        classifier_filter_builder=classifier_filter,
        random_generator=random_func,
        disturbance_type_map=disturbance_type_map,
    )

    return SITRuleBasedProcessor(
        event_processor,
        tr_processor,
        sit_events,
        sit_transitions,
        sit_eligibilities,
        reset_parameters,
    )
