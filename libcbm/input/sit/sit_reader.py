from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_disturbance_type_parser
from libcbm.input.sit import sit_age_class_parser
from libcbm.input.sit import sit_inventory_parser
from libcbm.input.sit import sit_yield_parser
from libcbm.input.sit import sit_disturbance_event_parser
from libcbm.input.sit import sit_transition_rule_parser
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from types import SimpleNamespace


def read(config):
    pass


def parse(sit_classifiers, sit_disturbance_types, sit_age_classes,
          sit_inventory, sit_yield, sit_events, sit_transitions):
    """Parses and validates CBM Standard import tool formatted data including
    the complicated interdependencies in the SIT format. Returns an object
    containing the validated result.

    The returned object has the following properties:

     - classifiers: a pandas.DataFrame of classifiers in the sit_classifiers
        input
     - classifier_values: a pandas.DataFrame of the classifier values in the
        sit_classifiers input
     - classifier_aggregates: a dictionary of the classifier aggregates
        in the sit_classifiers input
     - disturbance_types: a pandas.DataFrame based on the disturbance types in
        the sit_disturbance_types input
     - age_classes: a pandas.DataFrame of the age classes based on
        sit_age_classes
     - inventory: a pandas.DataFrame of the inventory based on sit_inventory
     - yield_table: a pandas.DataFrame of the merchantable volume yield curves
        in the sit_yield input
     - disturbance_events: a pandas.DataFrame of the disturbance events based
        on sit_events.  If the sit_events parameter is None this field is None.
     - transition_rules: a pandas.DataFrame of the transition rules based on
        sit_transitions.  If the sit_transitions parameter is None this field
        is None.

    Args:
        sit_classifiers (pandas.DataFrame): SIT formatted classifiers
        sit_disturbance_types (pandas.DataFrame): SIT formatted disturbance
            types
        sit_age_classes (pandas.DataFrame): SIT formatted age classes
        sit_inventory (pandas.DataFrame): SIT formatted inventory
        sit_yield (pandas.DataFrame): SIT formatted yield curves
        sit_events (pandas.DataFrame or None): SIT formatted disturbance events
        sit_transitions (pandas.DataFrame or None): SIT formatted transition
            rules

    Returns:
        object: an object containing parsed and validated SIT dataset
    """
    s = SimpleNamespace
    classifiers, classifier_values, classifier_aggregates = \
        sit_classifier_parser.parse(sit_classifiers)
    s.classifiers = classifiers
    s.classifier_values = classifier_values
    s.classifier_aggregates = classifier_aggregates
    s.disturbance_types = sit_disturbance_type_parser.parse(
        sit_disturbance_types)
    s.age_classes = sit_age_class_parser.parse(sit_age_classes)
    s.inventory = sit_inventory_parser.parse(
        sit_inventory, classifiers, classifier_values,
        s.disturbance_types, s.age_classes)
    s.yield_table = sit_yield_parser.parse(
        sit_yield, s.classifiers, s.classifier_values, s.age_classes)
    if sit_events:
        s.disturbance_events = sit_disturbance_event_parser.parse(
            sit_events, s.classifiers, s.classifier_values,
            s.classifier_aggregates, s.disturbance_types,
            s.age_classes)
    else:
        s.disturbance_events = None
    if sit_transitions:
        s.transition_rules = sit_transition_rule_parser.parse(
            sit_transitions, s.classifiers, s.classifier_values,
            s.classifier_aggregates, s.disturbance_types, s.age_classes)
    else:
        s.transition_rules = None
    return s
