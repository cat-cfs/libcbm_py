# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd

from libcbm.model.cbm.rule_based import event_processor
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target


class SITProcessEventsResult():

    def __init__(self, cbm_vars, stats_df, sit_disturbances):
        self.cbm_vars = cbm_vars
        self.stats_df = stats_df
        self.sit_disturbances = sit_disturbances

def get_pre_dynamics_func(sit_event_processor, sit_events):
    """Gets a function for applying SIT rule based events in a CBM
    timestep loop.

    The returned function can be used as the
    pre_dynamics_func argument of
    :py:func:`libcbm.model.cbm.cbm_simulator.simulate’`

    Args:
        sit_event_processor (SITEventProcessor): Instance of
            :py:class:`SITEventProcessor` for computing sit rule based
            disturbance events.
        sit_events (pandas.DataFrame): table of SIT formatted events.
            Expected format is the same as the return value of:
            :py:func:`libcbm.input.sit.sit_disturbance_event_parser.parse`
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
    def sit_events_pre_dynamics_func(time_step, cbm_vars, stats_func):
        cbm_vars, stats_df = sit_event_processor.process_events(
            time_step=time_step,
            sit_events=sit_events,
            cbm_vars=cbm_vars)
        stats_func(time_step, stats_df)
        return cbm_vars

    return sit_events_pre_dynamics_func


class SITEventProcessor():
    """SITEventProcessor processes standard import tool format events.

    Args:
        model_functions (libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper):
            The collection of CBM dynamics functions, which are used by this
            class to compute Carbon production due to disturbance events for
            MerchC targets.
        compute_functions (libcbm.wrapper.libcbm_wrapper.LibCBMWrapper): Used
            to compute carbon dynamics flows for MerchC targets.
        classifier_filter_builder (ClassifierFilter):
            object used to construct and evaluate classifier filters to
            include or exclude stands from event and transition eligibility
        random_generator (function): a function to generate a random sequence,
            whose single argument is an integer that specifies the number of
            random numbers in the returned sequence.
    """
    def __init__(self, model_functions, compute_functions,
                 classifier_filter_builder, random_generator):

        self.model_functions = model_functions
        self.compute_functions = compute_functions
        self.classifier_filter_builder = classifier_filter_builder
        self.random_generator = random_generator

    def _get_compute_disturbance_production(self, model_functions,
                                            compute_functions,
                                            eligible):

        def compute_disturbance_production(cbm_vars,
                                           disturbance_type_id):

            return rule_target.compute_disturbance_production(
                model_functions=model_functions,
                compute_functions=compute_functions,
                cbm_vars=cbm_vars,
                disturbance_type=disturbance_type_id,
                eligible=eligible)

        return compute_disturbance_production

    def _process_event(self, eligible, sit_event, sit_event_idx, cbm_vars):

        compute_disturbance_production = \
            self._get_compute_disturbance_production(
                model_functions=self.model_functions,
                compute_functions=self.compute_functions,
                eligible=eligible)

        target_factory = sit_stand_target.create_sit_event_target_factory(
            rule_target=rule_target,
            sit_event_row=sit_event,
            disturbance_production_func=compute_disturbance_production,
            random_generator=self.random_generator)

        pool_filter_expression, pool_filter_cols = \
            sit_stand_filter.create_pool_filter_expression(
                sit_event)
        state_filter_expression, state_filter_cols = \
            sit_stand_filter.create_state_filter_expression(
                sit_event, False)

        event_filter = rule_filter.merge_filters(
            rule_filter.create_filter(
                expression=pool_filter_expression,
                data=cbm_vars.pools,
                columns=pool_filter_cols),
            rule_filter.create_filter(
                expression=state_filter_expression,
                data=cbm_vars.state,
                columns=state_filter_cols),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    sit_event, cbm_vars.classifiers.columns.tolist()),
                cbm_vars.classifiers))

        stats_row = []

        def stats_func(event_stats):
            event_stats.update({"sit_event_index": sit_event_idx})
            stats_row.append(event_stats)

        cbm_vars = event_processor.process_event(
            event_filter=event_filter,
            undisturbed=eligible,
            target_func=target_factory,
            disturbance_type_id=sit_event["disturbance_type_id"],
            cbm_vars=cbm_vars,
            stats_func=stats_func)

        return cbm_vars, stats_row[0]

    def _event_iterator(self, sit_events):

        # TODO: In CBM-CFS3 events are sorted by default disturbance type id
        # (ascending) In libcbm, sort order needs to be explicitly defined in
        # cbm_defaults (or other place)
        sorted_events = sit_events.sort_values(
            by="disturbance_type_id",
            kind="mergesort")
        # mergesort is a stable sort, and the default "quicksort" is not
        for idx, sorted_event in sorted_events.iterrows():
            yield idx, dict(sorted_event)

    def process_events(self, time_step, sit_events, cbm_vars):
        """Process sit_events for the start of the given timestep, computing a
        new simulation state, and the disturbance types to apply for the
        timestep.

        Because of the nature of CBM rule based events, the size of the
        returned arrays may grow on the "n_stands" dimension from the original
        sizes due to area splitting, however the total inventory area will
        remain constant.

        Args:
            time_step (int): the simulation time step for which to compute the
                events.  Used to filter the specified sit_events DataFrame by
                its "time_step" column
            sit_events (pandas.DataFrame): table of SIT formatted events.
                Expected format is the same as the return value of:
                :py:func:`libcbm.input.sit.sit_disturbance_event_parser.parse`
            cbm_vars (object): an object containing dataframes that store cbm
                simulation state and variables

        Returns:
            object: expanded and updated cbm_vars

        """
        n_stands = cbm_vars.inventory.shape[0]
        cbm_vars.params.disturbance_type = np.zeros(n_stands, dtype=np.int32)

        time_step_events = sit_events[
            sit_events.time_step == time_step].copy()

        stats_rows = []
        for idx, sit_event in self._event_iterator(time_step_events):
            eligible = cbm_vars.params.disturbance_type <= 0
            cbm_vars, stats_row = self._process_event(
                eligible,
                sit_event,
                idx,
                cbm_vars)
            stats_rows.append(stats_row)

        stats_df = pd.DataFrame(stats_rows)
        return cbm_vars, stats_df
