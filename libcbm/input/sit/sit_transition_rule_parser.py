# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_classifier_parser

GROUPED_PERCENT_ERR_MAX = 0.00001


def parse(transition_rules, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):
    """Parses and validates the CBM SIT transition rule format.

    Args:
        transition_rules (pandas.DataFrame): CBM SIT transition rule formatted
            data.
        classifiers (pandas.DataFrame): used to validate the classifier
            set columns of the transition rule data. Use the return value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): used to validate the classifier
            set columns of the transition rule data. Use the return value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_aggregates (pandas.DataFrame): used to validate the
            classifier set columns of the transition rule data. Use the return
            value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        disturbance_types (pandas.DataFrame): Used to validate the
            disturbance_type column of the transition rule data. Use the return
            value of:
            :py:func:`libcbm.input.sit.sit_disturbance_types_parser.parse`
        age_classes (pandas.DataFrame): used to validate the number of volume
            columns.  Use the return value of:
            :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Raises:
        ValueError: undefined classifier values were found in the transition
            rule classifier sets
        ValueError: a grouped set of transition rules has a percent greater
            than 100%.
        ValueError: undefined disturbance types were found in the transition
            rule disturbance_type column

    Returns:
        pandas.DataFrame: validated transition rules
    """
    transition_rule_format = sit_format.get_transition_rules_format(
        classifiers.name, len(transition_rules.columns))

    transitions = sit_parser.unpack_table(
        transition_rules, transition_rule_format, "transitions")
    if len(transitions.index) == 0:
        return transitions
    # check that each value in transition_rules events classifier sets is
    # defined in classifier values, classifier aggregates or is a wildcard
    for row in classifiers.itertuples():
        source_classifiers = transitions[row.name].unique()

        # get the destination classifier
        tr_dest_fmt = sit_format.get_tr_classifier_set_postfix()
        dest_classifiers = transitions[f"{row.name}{tr_dest_fmt}"]

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

    parse_bool_func = sit_parser.get_parse_bool_func(
        "transitions", "using_age_class")
    transitions = sit_parser.substitute_using_age_class_rows(
        transitions, parse_bool_func, age_classes)

    # validate and substitute disturbance type names versus the SIT disturbance
    # types
    a = transitions.disturbance_type.unique()
    b = disturbance_types.id.unique()
    undefined_disturbances = np.setdiff1d(a, b)
    if len(undefined_disturbances) > 0:
        raise ValueError(
            "Undefined disturbance type ids (as defined in sit "
            f"disturbance types) detected: {undefined_disturbances}"
        )

    transitions = transitions.rename(
        columns={
            "min_softwood_age": "min_age",
            "max_softwood_age": "max_age"})

    transitions = transitions.drop(
        columns=["using_age_class", "min_hardwood_age", "max_hardwood_age"])

    # if the sum of percent for grouped transition rules exceeds 100% raise an
    # error
    group_cols = list(classifiers.name) + \
        ["min_age", "max_age", "disturbance_type"]
    if "spatial_reference" in transitions.columns: 
        group_cols += ["spatial_reference"]    
    grouped = transitions[group_cols + ["percent"]].groupby(group_cols).sum()
    invalid_grouped = grouped[
        grouped.percent > (100 + GROUPED_PERCENT_ERR_MAX)]
    if len(invalid_grouped) > 0:
        invalid_percents = [x.Index for x in invalid_grouped.head().itertuples()]
        raise ValueError(
            "the following groups have a total percent greater than 100%: "
            f"{invalid_percents}")

    return transitions
