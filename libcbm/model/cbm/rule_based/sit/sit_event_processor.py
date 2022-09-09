# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd

from libcbm.model.cbm.rule_based import event_processor
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target


class SITEventProcessor():
    """SITEventProcessor processes standard import tool format events.

    Args:
        cbm (object): CBM model
        classifier_filter_builder (ClassifierFilter):
            object used to construct and evaluate classifier filters to
            include or exclude stands from event and transition eligibility
        random_generator (function): a function to generate a random sequence,
            whose single argument is an integer that specifies the number of
            random numbers in the returned sequence.
    """
    def __init__(self, cbm, classifier_filter_builder, random_generator):

        self.cbm = cbm
        self.classifier_filter_builder = classifier_filter_builder
        self.random_generator = random_generator

    def _get_compute_disturbance_production(self, cbm, eligible):

        def compute_disturbance_production(cbm_vars, disturbance_type_id):

            return cbm.compute_disturbance_production(
                cbm_vars=cbm_vars,
                disturbance_type=disturbance_type_id,
                eligible=eligible)

        return compute_disturbance_production

    def _process_event(self, eligible, sit_event, cbm_vars,
                       sit_eligibility=None):

        compute_disturbance_production = \
            self._get_compute_disturbance_production(
                cbm=self.cbm,
                eligible=eligible)

        target_factory = sit_stand_target.create_sit_event_target_factory(
            rule_target=rule_target,
            sit_event_row=sit_event,
            disturbance_production_func=compute_disturbance_production,
            random_generator=self.random_generator)

        if sit_eligibility is None:
            event_filters = self._create_sit_event_filters(sit_event, cbm_vars)
        else:
            event_filters = [
                rule_filter.create_filter(
                    expression=sit_eligibility.pool_filter_expression,
                    data=cbm_vars.pools),
                rule_filter.create_filter(
                    expression=sit_eligibility.state_filter_expression,
                    data=cbm_vars.state),
                self.classifier_filter_builder.create_classifiers_filter(
                    sit_stand_filter.get_classifier_set(
                        sit_event, cbm_vars.classifiers.columns.tolist()),
                    cbm_vars.classifiers)]

        process_event_result = event_processor.process_event(
            event_filters=event_filters,
            undisturbed=eligible,
            target_func=target_factory,
            disturbance_type_id=sit_event["disturbance_type_id"],
            cbm_vars=cbm_vars)

        return process_event_result

    def _create_sit_event_filters(self, sit_event, cbm_vars):
        pool_filter_expression = \
            sit_stand_filter.create_pool_filter_expression(
                sit_event)
        state_filter_expression = \
            sit_stand_filter.create_state_filter_expression(
                sit_event, False)
        dist_type_filter_expression = \
            sit_stand_filter.create_last_disturbance_type_filter(
                sit_event)

        return [
            rule_filter.create_filter(
                expression=pool_filter_expression,
                data=cbm_vars.pools),
            rule_filter.create_filter(
                expression=state_filter_expression,
                data=cbm_vars.state),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    sit_event, cbm_vars.classifiers.columns.tolist()),
                cbm_vars.classifiers),
            rule_filter.create_filter(
                expression=dist_type_filter_expression,
                data=cbm_vars.state)]

    def _event_iterator(self, sit_events):
        sorted_events = sit_events.sort_values(
            by="sort_field",
            kind="mergesort")
        # mergesort is a stable sort, and the default "quicksort" is not
        for event_index, sorted_event in sorted_events.iterrows():
            yield event_index, dict(sorted_event)

    def process_events(self, time_step, sit_events, cbm_vars,
                       sit_eligibilities=None):
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
            sit_eligibilities (pandas.DataFrame): table of eligibility
                expressions with foreign key "disturbance_eligibility_id"

        Returns:
            object: expanded and updated cbm_vars

        """

        time_step_events = sit_events[
            sit_events.time_step == time_step].copy()

        stats_rows = []
        eligibilty_expressions = None
        if sit_eligibilities is not None:
            eligibilty_expressions = {
                int(row.disturbance_eligibility_id): row
                for _, row in sit_eligibilities.iterrows()
            }
        for event_index, sit_event in self._event_iterator(time_step_events):
            eligible = cbm_vars.parameters.disturbance_type <= 0
            expression = None
            if eligibilty_expressions:
                expression = eligibilty_expressions[
                    int(sit_event["disturbance_eligibility_id"])]
            process_event_result = self._process_event(
                eligible, sit_event, cbm_vars, expression)
            stats = process_event_result.rule_target_result.statistics
            if stats is not None:
                stats["sit_event_index"] = event_index
                stats_rows.append(stats)
        stats_df = pd.DataFrame(stats_rows) if stats is not None else None
        return cbm_vars, stats_df
