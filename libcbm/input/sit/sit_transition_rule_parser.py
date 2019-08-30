from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def parse(transition_rules, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):

    transition_rule_format = sit_format.get_transition_rules_format(
        classifiers.name, len(transition_rules.columns))

    unpacked_table = sit_parser.unpack_table(
        transition_rules, transition_rule_format, "yield")
    # check that the correct number of classifiers are present

    # check that each value in disturbance events classifier sets is defined in
    # classifier values, classifier aggregates or is a wildcard

    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output

    # check that all age criteria are identical between SW and HW (since CBM
    # has only a stand age)

    # translate criteria into a more useful format
    return unpacked_table
