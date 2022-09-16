# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd
from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def get_sort_types() -> dict[int, str]:
    """
    Gets the CBM standard import tool sorting id/name pairs as a dictionary
    """
    return {
        1: "PROPORTION_OF_EVERY_RECORD",
        2: "MERCHCSORT_TOTAL",
        3: "SORT_BY_SW_AGE",
        5: "SVOID",
        6: "RANDOMSORT",
        7: "TOTALSTEMSNAG",
        8: "SWSTEMSNAG",
        9: "HWSTEMSNAG",
        10: "MERCHCSORT_SW",
        11: "MERCHCSORT_HW",
        12: "SORT_BY_HW_AGE",
    }


def get_target_types() -> dict[str, str]:
    """Gets the CBM standard import tool target type id/name pairs as a
    dictionary
    """
    return {"A": "Area", "P": "Proportion", "M": "Merchantable"}


def parse_eligibilities(
    disturbance_events: pd.DataFrame, disturbance_eligibilities: pd.DataFrame
) -> pd.DataFrame:
    """Parse and validate disturbance eligibilities which are a libcbm-specific
    alternative to the eligibility columns in the cbm-cfs3 sit_disturbance
    events input.

    The benefit of this format is that the number of columns in sit_events is
    greatly reduced, and arbitrary boolean expressions of stand pool and state
    values, rather than min/max ranges supported in the CBM3-SIT format may be
    used.

    Example disturbance_eligibilities table:

     ==   =====================================  =======================
     id   pool_filter_expression                 state_filter_expression
     ==   =====================================  =======================
     1    (SoftwoodMerch + HardwoodMerch) >= 10  NULL
     2    (SoftwoodMerch + HardwoodMerch) >= 10  (age > 5) & (age < 100)
     3    NULL                                   NULL
     ==   =====================================  =======================

    * The id field in the disturbance_eligibilities corresponds to sit events
    * expressions are parsed by the numexpr library
    * note brackets are required around nested boolean expressions
      joined by a boolean operator (eg &)
    * for both pool_filter_expression, and state_filter_expression,
      the expressions must evaluate to a True or False value.  False
      indicates that the stand records being evaluated for the
      corresponding disturbance event deemed ineligible for the
      disturbance. True indicates that the expressions does not
      eliminate the stand from eligibility.
    * for pool_filter_expression any CBM pool is acceptable.  The pool names
      are defined in the cbm_defaults database tables.
    * for state_filter_expression any of the state values may be used in the
      boolean expression. See:
     :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`

    The final eligibility is evaluated as follows:

     ====================== ======================= =================
     pool_filter_expression state_filter_expression deemed_ineligible
     ====================== ======================= =================
     NULL or TRUE           NULL or TRUE            FALSE
     NULL or TRUE           FALSE                   TRUE
     FALSE                  NULL or TRUE            TRUE
     FALSE                  FALSE                   TRUE
     ====================== ======================= =================

    Args:
        disturbance_events (pandas.DataFrame): alternate form of CBM-CFS3
            sit_events: the 21 eligibility columns and the using age class
            and min-max columns are omitted.
        disturbance_eligibilities (pandas.DataFrame): table of id (int),
            state_filter expression (str), pool filter expression (str).
            The disturbance event disturbance_eligibility_id column
            corresponds to the id column in this table.

    Raises:
        ValueError: disturbance_eligibility_id values found in the specified
            sit_events were not present in the provided
            disturbance_eligibilities table.
        ValueError: at lease one null id value was detected in the id column
            of the specified disturbance_eligibilities table.
        ValueError: duplicate id value was detected in the id column of the
            specified disturbance_eligibilities table.

    Returns:
        pandas.DataFrame: the validated event eligibilities table
    """
    disturbance_eligibility_format = (
        sit_format.get_disturbance_eligibility_format()
    )

    eligibilities = sit_parser.unpack_table(
        disturbance_eligibilities,
        disturbance_eligibility_format,
        "disturbance eligibilities",
    )

    # confirm that each row in the disturbance events with an
    # eligibility id >= 0 has a corresponding record in the eligibilities
    # table
    missing_ids = set(disturbance_events["disturbance_eligibility_id"]) - set(
        eligibilities["disturbance_eligibility_id"]
    )
    if missing_ids:
        raise ValueError(
            "disturbance_eligibility_id values found in sit_events "
            f"but not in sit_disturbance_eligibilities {missing_ids}"
        )
    if pd.isnull(eligibilities.disturbance_eligibility_id).any():
        raise ValueError(
            "null values detected in eligibilities disturbance_eligibility_id "
            "column"
        )
    if eligibilities.disturbance_eligibility_id.duplicated().any():
        raise ValueError(
            "duplicated disturbance_eligibility_id values detected in "
            "eligibilities"
        )
    eligibilities = eligibilities.fillna("")
    return eligibilities


def parse(
    disturbance_events: pd.DataFrame,
    classifiers: pd.DataFrame,
    classifier_values: pd.DataFrame,
    classifier_aggregates: pd.DataFrame,
    disturbance_types: pd.DataFrame,
    age_classes: pd.DataFrame = None,
    separate_eligibilities: pd.DataFrame = False,
) -> pd.DataFrame:
    """Parses and validates the CBM SIT disturbance event format, or
    optionally an extended sit disturbance event format where disturbance
    eligibilites are separate from sit_events and joined by foreign key.

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
        age_classes (pandas.DataFrame, optional): used to validate and compute
            age eligibility criteria in disturbance_events. Use the return
            value of:
            :py:func:`libcbm.input.sit.sit_age_class_parser.parse`.
        disturbance_eligibilities (pandas.DataFrame, optional): table of
            eligibility expressions.

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
        classifiers.name,
        len(disturbance_events.columns),
        include_eligibility_columns=not separate_eligibilities,
    )

    events = sit_parser.unpack_table(
        disturbance_events, disturbance_event_format, "disturbance events"
    )

    # check that the correct number of classifiers are present, and check
    # that each value in disturbance events classifier sets is defined in
    # classifier values, classifier aggregates or is a wildcard
    for row in classifiers.itertuples():
        event_classifiers = events[row.name].unique()

        defined_classifiers = classifier_values[
            classifier_values["classifier_id"] == row.id
        ]["name"].unique()

        aggregates = np.array(
            [
                x["name"]
                for x in classifier_aggregates
                if x["classifier_id"] == row.id
            ]
        )
        wildcard = np.array([sit_classifier_parser.get_wildcard_keyword()])
        valid_classifiers = np.concatenate(
            [defined_classifiers, aggregates, wildcard]
        )

        diff_classifiers = np.setdiff1d(event_classifiers, valid_classifiers)
        if len(diff_classifiers) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_classifiers}"
            )

    if not separate_eligibilities:
        # if age classes are used substitute the age critera based on the age
        # class id, and raise an error if the id is not defined, and drop
        # using_age_class from output
        parse_bool_func = sit_parser.get_parse_bool_func(
            "events", "using_age_class"
        )
        events = sit_parser.substitute_using_age_class_rows(
            events, parse_bool_func, age_classes
        )
        events = events.rename(
            columns={
                "min_softwood_age": "min_age",
                "max_softwood_age": "max_age",
            }
        )

        events = events.drop(
            columns=["using_age_class", "min_hardwood_age", "max_hardwood_age"]
        )

    # validate sort type
    valid_sort_types = get_sort_types().keys()
    int_sort_type = events.sort_type.astype(int)
    sort_type_diff = set(int_sort_type.unique()).difference(
        set(valid_sort_types)
    )
    if len(sort_type_diff) > 0:
        raise ValueError(
            f"specified sort types are not valid: {sort_type_diff}"
        )
    events.sort_type = int_sort_type.map(get_sort_types())

    # validate target type
    valid_target_types = get_target_types().keys()
    target_type_diff = set(events.target_type.unique()).difference(
        set(valid_target_types)
    )
    if len(target_type_diff) > 0:
        raise ValueError(
            f"specified target types are not valid: {target_type_diff}"
        )
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

    return events
