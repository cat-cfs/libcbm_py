import numpy as np
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_classifier_parser


def parse(transition_rules, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):

    transition_rule_format = sit_format.get_transition_rules_format(
        classifiers.name, len(transition_rules.columns))

    unpacked_transitions = sit_parser.unpack_table(
        transition_rules, transition_rule_format, "yield")

    # check that each value in transition_rules events classifier sets is
    # defined in classifier values, classifier aggregates or is a wildcard
    for row in classifiers.itertuples():
        source_classifiers = unpacked_transitions[row.name].unique()

        # get the destination classifier
        tr_dest_fmt = sit_format.get_transition_rule_classifier_set_postfix()
        dest_classifiers = unpacked_transitions[f"{row.name}{tr_dest_fmt}"]

        defined_classifiers = classifier_values[
            classifier_values["classifier_id"] == row.id]["name"].unique()

        aggregates = np.array(
            [x["name"] for x in
             classifier_aggregates if x["classifier_id"] == row.id])
        wildcard = np.array([sit_classifier_parser.get_wildcard_keyword()])
        valid_source_classifiers = np.concatenate(
            [defined_classifiers, aggregates, wildcard])

        diff_source = np.setdiff1d(
            source_classifiers, valid_source_classifiers)
        if len(diff_source) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_source}")

        # aggregates may not appear in transition rule destination classifier
        # set (only the defined classifier values, or wildcards)
        valid_dest_classifiers = np.concatenate(
            [defined_classifiers, wildcard])
        diff_dest = np.setdiff1d(
            dest_classifiers, valid_dest_classifiers)
        if len(diff_dest) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_dest}")

    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output
    unpacked_transitions.using_age_class = \
        unpacked_transitions.using_age_class.map(
            sit_parser.get_parse_bool_func("transitions", "using_age_class"))
    using_age_class_rows = unpacked_transitions.loc[
        unpacked_transitions.using_age_class].copy()

    # check that all age criteria are identical between SW and HW (since CBM
    # has only a stand age)

    # validate and subsitute disturbance type names from the SIT disturbance
    # types
    undefined_disturbances = np.setdiff1d(
        unpacked_transitions.disturbance_type.unique(),
        disturbance_types.id.unique()
    )
    if len(undefined_disturbances) > 0:
        raise ValueError(
            "Undefined disturbance type ids (as defined in sit "
            f"disturbance types) detected: {undefined_disturbances}"
        )
    disturbance_join = unpacked_transitions.merge(
        disturbance_types, left_on="disturbance_type",
        right_on="id")
    unpacked_transitions.disturbance_type = disturbance_join.name

    # translate criteria into a more useful format

    return unpacked_transitions
