import numpy as np

from libcbm.model.cbm.rule_based import event_processor
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target

from libcbm.model.cbm import cbm_variables


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
    def sit_events_pre_dynamics_func(time_step, cbm_vars):

        classifiers, inventory = cbm_variables.inventory_to_df(
            cbm_vars.inventory)

        (disturbance_types,
         _classifiers,
         _inventory,
         _pools,
         _state) = sit_event_processor.process_events(
             time_step=time_step,
             sit_events=sit_events,
             classifiers=classifiers,
             inventory=inventory,
             pools=cbm_vars.pools,
             state_variables=cbm_vars.state)

        n_stands = _inventory.shape[0]
        cbm_vars.params = cbm_variables.initialize_cbm_parameters(
            n_stands=n_stands,
            disturbance_type=disturbance_types)

        cbm_vars.inventory = cbm_variables.initialize_inventory(
            classifiers=_classifiers,
            inventory=_inventory)

        cbm_vars.pools = _pools
        cbm_vars.state = _state

        # re-size the flux indicators array according to the number of stands
        cbm_vars.flux_indicators = cbm_variables.initialize_flux(
            n_stands=n_stands,
            flux_indicator_codes=list(cbm_vars.flux_indicators))

        return cbm_vars

    return sit_events_pre_dynamics_func


class SITEventProcessor():

    def __init__(self, model_functions,
                 compute_functions, cbm_defaults_ref,
                 classifier_filter_builder, random_generator,
                 on_unrealized_event):

        self.model_functions = model_functions
        self.compute_functions = compute_functions

        self.classifier_filter_builder = classifier_filter_builder
        self.cbm_defaults_ref = cbm_defaults_ref
        self.random_generator = random_generator

        self.on_unrealized_event = on_unrealized_event

    def _get_compute_disturbance_production(self, model_functions,
                                            compute_functions,
                                            eligible, flux_codes):

        def compute_disturbance_production(pools, inventory,
                                           disturbance_type_id):

            return rule_target.compute_disturbance_production(
                model_functions=model_functions,
                compute_functions=compute_functions,
                pools=pools,
                inventory=inventory,
                disturbance_type=disturbance_type_id,
                flux=cbm_variables.initialize_flux(
                    inventory.shape[0], flux_codes),
                eligible=eligible)

        return compute_disturbance_production

    def _process_event(self, eligible, sit_event, classifiers, inventory,
                       pools, state_variables):

        compute_disturbance_production = \
            self._get_compute_disturbance_production(
                model_functions=self.model_functions,
                compute_functions=self.compute_functions,
                eligible=eligible,
                flux_codes=self.cbm_defaults_ref.get_flux_indicators())

        # helper to indicate unrealized events to class user
        def on_unrealized(shortfall):
            self.on_unrealized_event(shortfall, sit_event)

        target_factory = sit_stand_target.create_sit_event_target_factory(
            rule_target=rule_target,
            sit_event_row=sit_event,
            disturbance_production_func=compute_disturbance_production,
            random_generator=self.random_generator,
            on_unrealized=on_unrealized)

        pool_filter_expression, pool_filter_cols = \
            sit_stand_filter.create_pool_filter_expression(
                sit_event)
        state_filter_expression, state_filter_cols = \
            sit_stand_filter.create_state_filter_expression(
                sit_event, False)

        event_filter = rule_filter.merge_filters(
            rule_filter.create_filter(
                expression=pool_filter_expression,
                data=pools,
                columns=pool_filter_cols),
            rule_filter.create_filter(
                expression=state_filter_expression,
                data=state_variables,
                columns=state_filter_cols),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    sit_event, classifiers.columns.tolist()),
                classifiers))

        return event_processor.process_event(
            filter_evaluator=rule_filter.evaluate_filter,
            event_filter=event_filter,
            undisturbed=eligible,
            target_func=target_factory,
            classifiers=classifiers,
            inventory=inventory,
            pools=pools,
            state_variables=state_variables)

    def _event_iterator(self, time_step, sit_events):

        # TODO: In CBM-CFS3 events are sorted by default disturbance type id
        # (ascending) In libcbm, sort order needs to be explicitly defined in
        # cbm_defaults (or other place)
        time_step_events = sit_events[
            sit_events.time_step == time_step].copy()

        time_step_events = time_step_events.sort_values(
            by="disturbance_type_id",
            kind="mergesort")
            # mergesort is a stable sort, and the default "quicksort" is not
        for _, time_step_event in time_step_events.iterrows():
            yield dict(time_step_event)

    def process_events(self, time_step, sit_events, classifiers, inventory,
                       pools, state_variables):
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
            classifiers (pandas.DataFrame): dataframe of classifier
                value ids by stand (row), by classifier (columns).  Column
                labels are the classifier names.
            inventory (pandas.DataFrame): table of inventory data by
                stand (rows)
            pools (pandas.DataFrame): CBM pool values by stand (rows), by
                classifier (columns).  Column labels are the pool names.
            state_variables (pandas.DataFrame): table of CBM simulation state
                variables by stand (rows)

        Returns:
            tuple: a tuple of the array of disturbance types for the specified
                events and simulation state, and the modified simulation state.

                1. disturbance_types (numpy.ndarray): array of CBM disturbance
                   type ids for each inventory index
                2. classifiers (pandas.DataFrame): updated CBM classifier
                   values
                3. inventory (pandas.DataFrame): updated CBM inventory
                4. pools (pandas.DataFrame): updated CBM simulation pools
                5. state_variables (pandas.DataFrame): updated CBM simulation
                   state variables

        """
        disturbance_types = np.zeros(inventory.shape[0], dtype=np.int32)
        eligible = np.ones(inventory.shape[0], dtype=bool)
        _classifiers = classifiers
        _inventory = inventory
        _pools = pools
        _state_variables = state_variables
        for sit_event in self._event_iterator(time_step, sit_events):
            target, _classifiers, _inventory, _pools, _state_variables = \
                self._process_event(
                    eligible,
                    sit_event,
                    _classifiers,
                    _inventory,
                    _pools,
                    _state_variables)

            target_area_proportions = target["area_proportions"]

            n_splits = (target_area_proportions < 1.0).sum()
            # update eligible to false at the disturbed indices, since they are
            # not eligible for the next event in this timestep.
            eligible[target["disturbed_index"]] = 0

            # extend the eligible array by the number of splits
            eligible = np.concatenate([eligible, np.ones(n_splits)])

            # set the disturbance types for the disturbed indices, based on
            # the sit_event disturbance_type field.
            disturbance_types[target["disturbed_index"]] = \
                sit_event["disturbance_type_id"]

            # extend the disturbance type array by the number of splits
            disturbance_types = np.concatenate(
                [disturbance_types, np.zeros(n_splits, dtype=np.int32)])

        return (
            disturbance_types, _classifiers, _inventory, _pools,
            _state_variables)
