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

    def _process_event(self, eligible, sit_event, cbm_vars):

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

        process_event_result = event_processor.process_event(
            event_filter=event_filter,
            undisturbed=eligible,
            target_func=target_factory,
            disturbance_type_id=sit_event["disturbance_type_id"],
            cbm_vars=cbm_vars)

        return process_event_result

    def _event_iterator(self, sit_events):

        # TODO: In CBM-CFS3 events are sorted by default disturbance type id
        # (ascending) In libcbm, sort order needs to be explicitly defined in
        # cbm_defaults (or other place)
        sorted_events = sit_events.sort_values(
            by="disturbance_type_id",
            kind="mergesort")
        # mergesort is a stable sort, and the default "quicksort" is not
        for event_index, sorted_event in sorted_events.iterrows():
            yield event_index, dict(sorted_event)

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
        for event_index, sit_event in self._event_iterator(time_step_events):
            eligible = cbm_vars.params.disturbance_type <= 0
            process_event_result = self._process_event(
                eligible, sit_event, cbm_vars)
            stats = process_event_result.rule_target_result.statistics
            stats["sit_event_index"] = event_index
            stats_rows.append(stats)

        stats_df = pd.DataFrame(stats_rows)
        return cbm_vars, stats_df
