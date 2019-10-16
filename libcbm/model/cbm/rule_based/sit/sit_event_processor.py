
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based import rule_target
from libcbm.model.cbm.rule_based import event_processor
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter

from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based.sit import sit_stand_target

from libcbm.model.cbm import cbm_variables


def get_compute_disturbance_production(model_functions, compute_functions,
                                       eligible, flux_codes,
                                       disturbance_type):

    def compute_disturbance_production(pools, inventory,
                                       disturbance_type_name):
        rule_target.compute_disturbance_production(
            model_functions=model_functions,
            compute_functions=compute_functions,
            pools=pools,
            inventory=inventory,
            disturbance_type=disturbance_type,
            flux=cbm_variables.initialize_flux(
                inventory.shape[0], flux_codes),
            eligible=eligible
            )
    return compute_disturbance_production


def event_iterator(time_step, sit_events):

    # TODO: In CBM-CFS3 events are sorted by default disturbance type id
    # (ascending) In libcbm, sort order needs to be explicitly defined in
    # cbm_defaults (or other place)
    time_step_events = sit_events[
        sit_events.time_step == time_step]
    for _, time_step_event in time_step_events.itterows():
        yield dict(time_step_event)


class SITEventProcessor():
    def __init__(self, model_functions, compute_functions, cbm_defaults_ref,
                 classifiers_config, classifier_aggregates, random_generator):
        self.model_functions = model_functions
        self.compute_functions = compute_functions

        self.classifier_filter_builder = ClassifierFilter(
            classifiers_config, classifier_aggregates)
        self.cbm_defaults_ref = cbm_defaults_ref
        self.random_generator = random_generator

    def process_event(self, eligible, sit_event, classifiers, inventory, pools,
                      state_variables, on_unrealized):
        pool_filter, pool_filter_cols = \
            sit_stand_filter.create_pool_value_filter_expression(
                sit_event)
        state_filter, state_filter_cols = \
            sit_stand_filter.create_state_variable_filter_expression(
                sit_event,
                sit_stand_filter.get_state_variable_age_filter_mappings())
        classifier_filter = \
            self.classifier_filter_builder.create_classifiers_filter(
                sit_stand_filter.get_classifier_set(
                    sit_event, classifiers.columns.tolist()),
                classifiers)
        compute_disturbance_production = get_compute_disturbance_production(
            model_functions=self.model_functions,
            compute_functions=self.compute_functions,
            eligible=eligible,
            flux_codes=self.cbm_defaults_ref.get_flux_indicators(),
            disturbance_types=self.cbm_defaults_ref.get_disturbance_type_id(
                sit_event["disturbance_type"])
        )
        target_factory = sit_stand_target.create_sit_event_target_factory(
            rule_target=rule_target,
            sit_event_row=sit_event,
            disturbance_production_func=compute_disturbance_production,
            eligible=eligible,
            random_generator=self.random_generator,
            on_unrealized=on_unrealized)

        rule_filter.create_filter(
            pool_filter, pools, pool_filter_cols)
        rule_filter.create_filter(
            state_filter, state_variables, state_filter_cols)
        event_processor.process_event(
            filter_factory=rule_filter,
            classifiers_filter_factory=self.classifier_filter_builder,
            filter_data=
            )