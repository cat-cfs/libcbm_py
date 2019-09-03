from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def parse(disturbance_events, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):

    disturbance_event_format = sit_format.get_disturbance_event_format(
          classifiers.name, len(disturbance_events.columns))

    unpacked_table = sit_parser.unpack_table(
        disturbance_events, disturbance_event_format, "disturbance events")

    # check that the correct number of classifiers are present

    # check that each value in disturbance events classifier sets is defined in
    # classifier values, classifier aggregates or is a wildcard

    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output

    # check that all age criteria are identical between SW and HW (since CBM
    # has only a stand age)

    # validate sort type

    # validate target type

    # validate disturbance type according to specified disturbance types
    return unpacked_table
