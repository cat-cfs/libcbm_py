from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def parse(disturbance_events, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):

    disturbance_event_format = sit_format.get_disturbance_event_format(
          classifiers.name, len(disturbance_events.columns))

    unpacked_table = sit_parser.unpack_table(
        disturbance_events, disturbance_event_format, "disturbance events")


    return unpacked_table




