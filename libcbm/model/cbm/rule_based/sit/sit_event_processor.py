from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target

from libcbm.model.cbm import cbm_variables


def event_iterator(time_step, sit_events):

    # TODO: In CBM-CFS3 events are sorted by default disturbance type id
    # (ascending) In libcbm, sort order needs to be explicitly defined in
    # cbm_defaults (or other place)
    time_step_events = sit_events[
        sit_events.time_step == time_step]
    for _, time_step_event in time_step_events.itterows():
        yield dict(time_step_event)


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
                eligible=eligible
                )
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

        _classifiers, _inventory, _pools, _state_variables = \
            self.event_processor.process_event(
                filter_evaluator=self.rule_filter.evaluate_filter,
                event_filter=event_filter,
                undisturbed=eligible,
                target_func=target_factory,
                classifiers=classifiers,
                inventory=inventory,
                pools=pools,
                state_variables=state_variables)


    def process_events(self, time_step, sit_events, classifiers, inventory,
                       pools, state_variables, on_unrealized):
        for sit_event in event_iterator(time_step, sit_events):
            _classifiers, _inventory, _pools, _state_variables = \
                self.process_event(
                    eligible,
                    sit_event,
                    classifiers,
                    inventory,
                    pools,
                    state_variables,
                    on_unrealized)
