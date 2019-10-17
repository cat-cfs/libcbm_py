import pandas as pd
import numpy as np
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target

from libcbm.model.cbm import cbm_variables


class SITEventProcessor():

    def __init__(self, rule_filter_functions, rule_target_functions,
                 event_processor_functions, model_functions,
                 compute_functions, cbm_defaults_ref,
                 classifier_filter_builder, random_generator):

        self.model_functions = model_functions
        self.compute_functions = compute_functions

        self.classifier_filter_builder = classifier_filter_builder
        self.cbm_defaults_ref = cbm_defaults_ref
        self.random_generator = random_generator

        self.event_processor = event_processor_functions
        self.rule_filter = rule_filter_functions
        self.rule_target = rule_target_functions

        # this is for looking up disturbance type ids given a disturbance type
        # name.
        self.disturbance_type_id_lookup = {
            x["disturbance_type_name"]: x["disturbance_type_id"]
            for x in self.cbm_defaults_ref.get_disturbance_types()}

    def get_compute_disturbance_production(self, model_functions,
                                           compute_functions,
                                           eligible, flux_codes):

        def compute_disturbance_production(pools, inventory,
                                           disturbance_type):

            lookup_func = self.cbm_defaults_ref.get_disturbance_type_id
            self.rule_target.compute_disturbance_production(
                model_functions=model_functions,
                compute_functions=compute_functions,
                pools=pools,
                inventory=inventory,
                disturbance_type=lookup_func(disturbance_type),
                flux=cbm_variables.initialize_flux(
                    inventory.shape[0], flux_codes),
                eligible=eligible)

        return compute_disturbance_production

    def process_event(self, eligible, sit_event, classifiers, inventory, pools,
                      state_variables, on_unrealized):

        compute_disturbance_production = \
            self.get_compute_disturbance_production(
                model_functions=self.model_functions,
                compute_functions=self.compute_functions,
                eligible=eligible,
                flux_codes=self.cbm_defaults_ref.get_flux_indicators())

        target_factory = sit_stand_target.create_sit_event_target_factory(
            rule_target=self.rule_target,
            sit_event_row=sit_event,
            disturbance_production_func=compute_disturbance_production,
            eligible=eligible,
            random_generator=self.random_generator,
            on_unrealized=on_unrealized)

        pool_filter_expression, pool_filter_cols = \
            sit_stand_filter.create_pool_value_filter_expression(
                sit_event)
        state_filter_expression, state_filter_cols = \
            sit_stand_filter.create_state_variable_filter_expression(
                sit_event,
                sit_stand_filter.get_state_variable_filter_mappings())

        event_filter = self.rule_filter.merge_filters(
            self.rule_filter.create_filter(
                expression=pool_filter_expression,
                data=pools,
                columns=pool_filter_cols),
            self.rule_filter.create_filter(
                expression=state_filter_expression,
                data=state_variables,
                columns=state_filter_cols),
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    sit_event, classifiers.columns.tolist()),
                classifiers))

        return self.event_processor.process_event(
            filter_evaluator=self.rule_filter.evaluate_filter,
            event_filter=event_filter,
            undisturbed=eligible,
            target_func=target_factory,
            classifiers=classifiers,
            inventory=inventory,
            pools=pools,
            state_variables=state_variables)

    def event_iterator(self, time_step, sit_events):

        # TODO: In CBM-CFS3 events are sorted by default disturbance type id
        # (ascending) In libcbm, sort order needs to be explicitly defined in
        # cbm_defaults (or other place)
        time_step_events = sit_events[
            sit_events.time_step == time_step].copy()

        time_step_events.disturbance_type_id = \
            time_step_events.disturbance_type.map(
                self.disturbance_type_id_lookup)
        time_step_events.sort_values(by="disturbance_type_id")
        for _, time_step_event in time_step_events.itterows():
            yield dict(time_step_event)

    def process_events(self, time_step, sit_events, classifiers, inventory,
                       pools, state_variables, on_unrealized):

        disturbance_types = np.zeros(inventory.shape[0], dtype=np.int32)
        eligible = np.ones(inventory.shape[0])
        _classifiers = classifiers
        _inventory = inventory
        _pools = pools
        _state_variables = state_variables
        for sit_event in self.event_iterator(time_step, sit_events):
            target, _classifiers, _inventory, _pools, _state_variables = \
                self.process_event(
                    eligible,
                    sit_event,
                    _classifiers,
                    _inventory,
                    _pools,
                    _state_variables,
                    on_unrealized)

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

    def get_pre_dynamics_func(self, sit_events):

        def sit_events_pre_dynamics_func(time_step, cbm_vars):

            classifiers, inventory = cbm_variables.inventory_to_df(
                cbm_vars.inventory)

            (disturbance_types,
             _classifiers,
             _inventory,
             _pools,
             _state) = self.process_events(
                 time_step,
                 sit_events,
                 classifiers,
                 inventory,
                 cbm_vars.pools,
                 cbm_vars.state,
                 None)

            n_stands = _inventory.shape[0]
            cbm_vars.params = cbm_variables.initialize_cbm_parameters(
                n_stands=n_stands,
                disturbance_type=disturbance_types)

            cbm_vars.inventory = cbm_variables.initialize_inventory(
                classifiers=_classifiers,
                inventory=_inventory)

            cbm_vars.pools = _pools
            cbm_vars.state = _state
            return cbm_vars

        return sit_events_pre_dynamics_func
