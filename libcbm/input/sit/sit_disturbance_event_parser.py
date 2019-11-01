"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def get_sort_types():
    """Gets the CBM standard import tool sorting id/name pairs as a dictionary
    """
    return {
        1: "PROPORTION_OF_EVERY_RECORD",
        2: "MERCHCSORT_TOTAL",
        3: "SORT_BY_SW_AGE",
        5: "SVOID ",
        6: "RANDOMSORT",
        7: "TOTALSTEMSNAG",
        8: "SWSTEMSNAG",
        9: "HWSTEMSNAG",
        10: "MERCHCSORT_SW",
        11: "MERCHCSORT_HW",
        12: "SORT_BY_HW_AGE"}


def get_target_types():
    """Gets the CBM standard import tool target type id/name pairs as a
    dictionary
    """
    return {
        "A": "Area",
        "P": "Proportion",
        "M": "Merchantable"}


def parse(disturbance_events, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):
    """Parses and validates the CBM SIT disturbance event format.

    Args:
        disturbance_events (pandas.DataFrame): CBM SIT disturbance events
            formatted data.
        classifiers (pandas.DataFrame): used to validate the classifier
            set columns of the disturbance event data. Use the return value
            of: :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): used to validate the classifier
            set columns of the disturbance event data. Use the return value
            of: :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_aggregates (pandas.DataFrame): used to validate the
            classifier set columns of the disturbance event data. Use the
            return value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        disturbance_types (pandas.DataFrame): Used to validate the
            disturbance_type column of the disturbance event data. Use the
            return value of:
            :py:func:`libcbm.input.sit.sit_disturbance_types_parser.parse`
        age_classes (pandas.DataFrame): used to validate and compute age
            eligibility criteria in disturbance_events. Use the return value
            of: :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Raises:
        ValueError: undefined classifier values were found in the disturbance
            event classifier sets
        ValueError: undefined disturbance types were found in the disturbance
            event disturbance_type column
        ValueError: undefined sort types were found in the disturbance
            event sort_type column. See :py:func:`get_sort_types`
        ValueError: undefined target types were found in the disturbance
            event target_type column. See :py:func:`get_target_types`

    Returns:
        pandas.DataFrame: the validated disturbance events
    """
    disturbance_event_format = sit_format.get_disturbance_event_format(
          classifiers.name, len(disturbance_events.columns))

    events = sit_parser.unpack_table(
        disturbance_events, disturbance_event_format, "disturbance events")

    # check that the correct number of classifiers are present, and check
    # that each value in disturbance events classifier sets is defined in
    # classifier values, classifier aggregates or is a wildcard
    for row in classifiers.itertuples():
        event_classifiers = events[row.name].unique()

        defined_classifiers = classifier_values[
            classifier_values["classifier_id"] == row.id]["name"].unique()

        aggregates = np.array(
            [x["name"] for x in
             classifier_aggregates if x["classifier_id"] == row.id])
        wildcard = np.array([sit_classifier_parser.get_wildcard_keyword()])
        valid_classifiers = np.concatenate(
            [defined_classifiers, aggregates, wildcard])

        diff_classifiers = np.setdiff1d(
            event_classifiers, valid_classifiers)
        if len(diff_classifiers) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_classifiers}")

    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output
    parse_bool_func = sit_parser.get_parse_bool_func(
        "events", "using_age_class")
    events = sit_parser.substitute_using_age_class_rows(
        events, parse_bool_func, age_classes)

    # validate sort type
    valid_sort_types = get_sort_types().keys()
    int_sort_type = events.sort_type.astype(np.int)
    sort_type_diff = set(int_sort_type.unique()) \
        .difference(set(valid_sort_types))
    if len(sort_type_diff) > 0:
        raise ValueError(
            f"specified sort types are not valid: {sort_type_diff}")
    events.sort_type = int_sort_type.map(get_sort_types())

    # validate target type
    valid_target_types = get_target_types().keys()
    target_type_diff = set(events.target_type.unique()) \
        .difference(set(valid_target_types))
    if len(target_type_diff) > 0:
        raise ValueError(
            f"specified target types are not valid: {target_type_diff}")
    events.target_type = events.target_type.map(get_target_types())

    # validate disturbance type according to specified disturbance types
    a = events.disturbance_type.unique()
    b = disturbance_types.id.unique()
    undefined_disturbances = np.setdiff1d(a, b)
    if len(undefined_disturbances) > 0:
        raise ValueError(
            "Undefined disturbance type ids (as defined in sit "
            f"disturbance types) detected: {undefined_disturbances}"
        )
    disturbance_join = events.merge(
        disturbance_types, left_on="disturbance_type",
        right_on="id")
    events.disturbance_type = disturbance_join.name

    events = events.rename(
        columns={
            "min_softwood_age": "min_age",
            "max_softwood_age": "max_age"})

    events = events.drop(
        columns=["using_age_class", "min_hardwood_age", "max_hardwood_age"])
    return events
