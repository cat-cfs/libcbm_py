# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
import re
import pandas as pd


def get_tr_classifier_set_postfix() -> str:
    """since transition rules contain 2 classifier sets (2 sets of columns)
    duplicate names are a problem if the classifier names are used for both.
    This function returns a postfix to append onto the second set of
    classifiers to solve that issue.
    """
    return "_tr"


def adjust_classifier_name(classifier_name: str) -> str:
    if classifier_name.isidentifier():
        return classifier_name
    else:
        return re.sub(r"\W|^(?=\d)", "_", classifier_name)


def adjust_classifier_names(classifier_names: pd.Series) -> pd.Series:
    """Make each of the classifier names in the specified series valid python
    identifiers

    Args:
        classifier_names (pd.Series): the unadjusted classifier names from the
            SIT format.

    Returns:
        pd.Series: adjusted series of classifier names
    """
    classifier_name_list = [str(c) for c in classifier_names]
    adjusted_name_list = []
    for c in classifier_name_list:
        adjusted_name_list.append(adjust_classifier_name(c))
    return pd.Series(adjusted_name_list)


def get_classifier_format(n_columns: int) -> list[dict]:
    """Gets a list of dictionaries describing the CBM SIT classifier columns

    Args:
        n_columns (int): the number of columns in an sit classifiers formatted
            table

    Raises:
        ValueError: raised if the number of columns is less than the minimum
            required.

    Returns:
        list: a list of dictionaries describing the CBM SIT classifier columns
    """
    classifier_format = [
        {"name": "id", "index": 0, "type": int},
        {"name": "name", "index": 1, "type": str},
        {"name": "description", "index": 2, "type": str},
    ]

    if n_columns < 3:
        raise ValueError(
            "specified number of columns invalid.  Expected at least 3."
        )
    elif n_columns >= 4:
        classifier_format.extend(
            [
                {
                    "name": "aggregate_value_{}".format(i - 2),
                    "index": i,
                    "type": str,
                }
                for i in range(3, n_columns)
            ]
        )
    return classifier_format


def get_disturbance_type_format(n_columns: int) -> list[dict]:
    """Gets a list of dictionaries describing the CBM SIT disturbance type
    columns

    Args:
        n_columns (int): The number of columns in a SIT disturbance types
            formatted table.

    Raises:
        ValueError: n_columns is less than the minimum required number of
            columns for the SIT disturbance type format.
        ValueError: n_columns is more than the required number of columns for
            the sit disturbance type format.

    Returns:
        list: a list of dictionaries that describe the CBM SIT disturbance
            type columns
    """
    disturbance_type_format = [
        {"name": "id", "index": 0, "type": str},
        {"name": "name", "index": 1, "type": str},
    ]
    if n_columns < 2:
        raise ValueError(
            "specified number of columns invalid.  Expected at least 2."
        )
    elif n_columns == 3:
        disturbance_type_format.append({"name": "description", "index": 2})
    elif n_columns > 3:
        raise ValueError(
            "specified number of columns invalid.  Expected at most 3."
        )
    return disturbance_type_format


def get_age_class_format() -> list[dict]:
    """Gets a list of dictionaries describing the CBM SIT age class columns

    Returns:
        list: a list of dictionaries that describe the  CBM SIT age class
            columns
    """
    return [
        {"name": "id", "index": 0, "type": str},
        {"name": "class_size", "index": 1, "type": int},
    ]


def get_yield_format(
    classifier_names: list[str], n_columns: int
) -> list[dict]:
    """Gets a list of dictionaries describing the CBM SIT age class columns

    Args:
        classifier_names (list): a list of strings which are the names of the
            classifiers
        n_columns (int): The number of columns in a SIT yield formatted table.

    Raises:
        ValueError: the specified number of columns is less than the minimum
            number of columns for a valid SIT yield formatted table

    Returns:
        list: a list of dictionaries that describe the  CBM SIT yield table
            columns
    """
    n_classifiers = len(classifier_names)
    classifier_values = [
        {"name": c, "index": i, "type": str}
        for i, c in enumerate(classifier_names)
    ]
    leading_species_col = [{"name": "leading_species", "index": n_classifiers}]
    vol_index = n_classifiers + 1
    if n_columns < vol_index + 1:
        # in SIT vol_index is the 0th age volume, so at least 2 volumes need
        # to be here for anything to happen in CBM
        raise ValueError(
            "at least {0} columns are required".format(vol_index + 1)
        )
    volumes = [
        {
            "name": "v{}".format(i - vol_index),
            "index": i,
            "min_value": 0,
            "type": float,
        }
        for i in range(vol_index, n_columns)
    ]

    return classifier_values + leading_species_col + volumes


def get_age_eligibility_columns(base_index: int) -> list[dict]:
    """gets the columns for age eligibility which appear in SIT events and
    SIT transition.  The index of the columns is offset using the specified
    base.

    Args:
        base_index (index): the index of the first age eligibility columns

    Returns:
        list: a list of dictionaries describing the SIT age eligibility
            columns
    """
    return [
        {"name": "using_age_class", "index": base_index},
        {"name": "min_softwood_age", "index": base_index + 1},
        {"name": "max_softwood_age", "index": base_index + 2},
        {"name": "min_hardwood_age", "index": base_index + 3},
        {"name": "max_hardwood_age", "index": base_index + 4},
    ]


def get_transition_rules_format(
    classifier_names: list[str], n_columns: int
) -> list[dict]:
    """Generate a list of dictionaries describing each column in the SIT
    format transition rules.  The format is dynamic and changes based on the
    number of classifiers and whether or not a spatial identifier is
    specified.

    Args:
        classifier_names (list): a list of the names of classifiers
        n_columns (int): the number of columns in transition rules data.
            This is used to detect whether or not a spatial identifier is
            included in the data.

    Raises:
        ValueError: n_columns was not valid for the sit transitions format

    Returns:
        list: a list of dictionaries describing the SIT transition rule columns
    """
    n_classifiers = len(classifier_names)
    classifier_set_src = [
        {"name": c, "index": i, "type": str}
        for i, c in enumerate(classifier_names)
    ]
    age_eligibility = get_age_eligibility_columns(n_classifiers)
    disturbance_type = [
        {"name": "disturbance_type", "index": n_classifiers + 5, "type": str}
    ]

    regeneration_delay_index = 2 * n_classifiers + len(age_eligibility) + 1
    post_transition = [
        {
            "name": "regeneration_delay",
            "index": regeneration_delay_index,
            "min_value": 0,
            "type": int,
        },
        {
            "name": "reset_age",
            "index": regeneration_delay_index + 1,
            "min_value": -1,
            "type": int,
        },
        {
            "name": "percent",
            "index": regeneration_delay_index + 2,
            "min_value": 0,
            "max_value": 100,
            "type": float,
        },
    ]
    spatial_reference = [
        {
            "name": "spatial_reference",
            "index": regeneration_delay_index + 3,
            "type": int,
        }
    ]
    tr_cset_postfix = get_tr_classifier_set_postfix()
    classifier_set_dest = [
        {
            "name": f"{c}{tr_cset_postfix}",
            "index": i + n_classifiers + len(age_eligibility) + 1,
            "type": str,
        }
        for i, c in enumerate(classifier_names)
    ]
    result = []
    result.extend(classifier_set_src)  # source classifier set
    result.extend(age_eligibility)
    result.extend(disturbance_type)
    result.extend(classifier_set_dest)  # destination classifier set
    result.extend(post_transition)
    if n_columns < len(result):
        raise ValueError(
            "specified number of columns invalid.  Expected at least "
            "{}".format(len(result))
        )
    if n_columns == len(result):
        return result
    elif n_columns == len(result) + 1:
        result.extend(spatial_reference)
        return result
    else:
        raise ValueError(
            "incorrect number of columns in transition rules. "
            "Expected at most {}".format(len(result) + 1)
        )


def get_inventory_format(
    classifier_names: list[str], n_columns: int, has_inventory_ids: bool
) -> list[dict]:
    """Gets a description of the SIT inventory columns as a list of
    dictionaries

    Args:
        classifier_names (list): a list of the names of classifiers
        n_columns (int): the number of columns in inventory data.  This
            is required because the format has a varying number of optional
            columns.
        has_inventory_ids (bool): if true, the table is expected to have a
            leading column defining the inventory id for each row for
            simulation area tracking purposes

    Raises:
        ValueError: The number of columns was incorrect

    Returns:
        list: a list of dictionaries describing the SIT inventory columns
    """
    n_classifiers = len(classifier_names)
    n_leading_cols = n_classifiers

    inventory_id = []
    if has_inventory_ids:
        n_leading_cols += 1
        inventory_id.append({"name": "inventory_id", "index": 0, "type": int})
    classifier_set = [
        {"name": c, "index": i + len(inventory_id), "type": str}
        for i, c in enumerate(classifier_names)
    ]

    inventory = [
        {"name": "using_age_class", "index": n_leading_cols, "type": str},
        # age can be a string (for "using_age_class" support, so no min value
        # is specified)
        {"name": "age", "index": n_leading_cols + 1},
        {
            "name": "area",
            "index": n_leading_cols + 2,
            "min_value": 0,
            "type": float,
        },
        {
            "name": "delay",
            "index": n_leading_cols + 3,
            "min_value": 0,
            "type": int,
        },
        {"name": "land_class", "index": n_leading_cols + 4},
    ]

    if n_columns > n_leading_cols + 6:
        inventory.extend(
            [
                {
                    "name": "historical_disturbance_type",
                    "index": n_leading_cols + 5,
                    "type": str,
                },
                {
                    "name": "last_pass_disturbance_type",
                    "index": n_leading_cols + 6,
                    "type": str,
                },
            ]
        )
    if n_columns == n_leading_cols + 6:
        raise ValueError(
            "Invalid number of columns: both historical and last pass "
            "disturbance types must be defined."
        )
    if n_columns < n_leading_cols + 5:
        raise ValueError(
            f"With {n_leading_cols} classifiers, SIT inventory should have "
            f"at least {n_leading_cols + 5} columns."
        )
    if n_columns == n_leading_cols + 8:
        inventory.append(
            {
                "name": "spatial_reference",
                "index": n_leading_cols + 7,
                "type": int,
            }
        )
    if n_columns > n_leading_cols + 8:
        raise ValueError(
            f"With {n_leading_cols} classifiers, SIT inventory should have "
            f"at most {n_leading_cols + 8} columns."
        )

    return inventory_id + classifier_set + inventory


def get_disturbance_eligibility_columns(index: int) -> list[dict]:
    """gets the columns for disturbance eligibility which appear in SIT
    events.  The index of the columns is offset using the specified
    base.

    Args:
        base_index (index): the index of the first eligibility column

    Returns:
        list: a list of dictionaries describing the SIT disturbance
            eligibility columns
    """
    return [
        {"name": "MinYearsSinceDist", "index": index + 0, "type": float},
        {"name": "MaxYearsSinceDist", "index": index + 1, "type": float},
        {"name": "LastDistTypeID", "index": index + 2, "type": str},
        {"name": "MinTotBiomassC", "index": index + 3, "type": float},
        {"name": "MaxTotBiomassC", "index": index + 4, "type": float},
        {"name": "MinSWMerchBiomassC", "index": index + 5, "type": float},
        {"name": "MaxSWMerchBiomassC", "index": index + 6, "type": float},
        {"name": "MinHWMerchBiomassC", "index": index + 7, "type": float},
        {"name": "MaxHWMerchBiomassC", "index": index + 8, "type": float},
        {"name": "MinTotalStemSnagC", "index": index + 9, "type": float},
        {"name": "MaxTotalStemSnagC", "index": index + 10, "type": float},
        {"name": "MinSWStemSnagC", "index": index + 11, "type": float},
        {"name": "MaxSWStemSnagC", "index": index + 12, "type": float},
        {"name": "MinHWStemSnagC", "index": index + 13, "type": float},
        {"name": "MaxHWStemSnagC", "index": index + 14, "type": float},
        {"name": "MinTotalStemSnagMerchC", "index": index + 15, "type": float},
        {"name": "MaxTotalStemSnagMerchC", "index": index + 16, "type": float},
        {"name": "MinSWMerchStemSnagC", "index": index + 17, "type": float},
        {"name": "MaxSWMerchStemSnagC", "index": index + 18, "type": float},
        {"name": "MinHWMerchStemSnagC", "index": index + 19, "type": float},
        {"name": "MaxHWMerchStemSnagC", "index": index + 20, "type": float},
    ]


def get_disturbance_event_format(
    classifier_names: list[str],
    n_columns: int,
    include_eligibility_columns: bool = True,
    has_disturbance_event_ids: bool = False,
) -> list[dict]:
    """Gets a list of column description dictionaries describing the SIT
    disturbance event format

    Args:
        classifier_names (list): a list of the names of classifiers
        n_columns (int): the number of columns in disturbance data.  This
            is required because the format has a varying number of optional
            columns.
        include_eligibility_columns (bool, optional): if set to false the
            standard age eligibility and carbon eligibility columns are
            excluded from the result, and an eligibility_id
            column instead is included.

    Raises:
        ValueError: specified number of columns is invalid

    Returns:
        list: a list of dictionaries describing the SIT disturbance events
            table columns
    """
    n_classifiers = len(classifier_names)
    n_leading_cols = n_classifiers
    disturbance_event_id = []
    if has_disturbance_event_ids:
        n_leading_cols += 1
        disturbance_event_id.append(
            {"name": "disturbance_event_id", "index": 0, "type": int}
        )

    classifier_set = [
        {"name": c, "index": i + len(disturbance_event_id), "type": str}
        for i, c in enumerate(classifier_names)
    ]

    eligibiliy_cols = []
    if include_eligibility_columns:
        disturbance_age_eligibility = get_age_eligibility_columns(
            n_leading_cols
        )
        n_age_fields = len(disturbance_age_eligibility)
        disturbance_eligibility = get_disturbance_eligibility_columns(
            n_leading_cols + n_age_fields
        )
        n_eligibility_fields = len(disturbance_eligibility)
        index = n_leading_cols + n_age_fields + n_eligibility_fields
        eligibiliy_cols.extend(disturbance_age_eligibility)
        eligibiliy_cols.extend(disturbance_eligibility)
    else:
        eligibiliy_cols.append(
            {
                "name": "eligibility_id",
                "index": n_leading_cols,
                "type": int,
            }
        )
        index = n_leading_cols + 1
    event_target = [
        {
            "name": "efficiency",
            "index": index,
            "type": float,
            "min_value": 0,
            "max_value": 1,
        },
        {"name": "sort_type", "index": index + 1},
        {"name": "target_type", "index": index + 2},
        {"name": "target", "index": index + 3, "type": float, "min_value": 0},
        {"name": "disturbance_type", "index": index + 4, "type": str},
        {"name": "time_step", "index": index + 5, "type": int, "min_value": 1},
    ]
    if n_columns < index + 6:
        raise ValueError(
            "specified number of columns invalid.  Expected at least "
            "{}".format(index + 6)
        )
    if n_columns == index + 7:
        event_target.append(
            {"name": "spatial_reference", "index": index + 6, "type": int}
        )
    if n_columns > index + 7:
        raise ValueError(
            "specified number of columns invalid.  Expected at most "
            "{}".format(index + 7)
        )
    return (
        disturbance_event_id + classifier_set + eligibiliy_cols + event_target
    )


def get_eligibility_format(ncols: int) -> list[dict]:
    if ncols < 5:
        raise ValueError("number of columns must be > 5")
    fmt = [
        {"name": "eligibility_id", "index": 0, "type": int},
        {"name": "description", "index": 1, "type": str},
        {"name": "expression_type", "index": 2, "type": str},
        {"name": "expression", "index": 3, "type": str},
    ]
    for i in range(4, ncols):
        p_idx = i - 3
        fmt.append({"name": f"p{p_idx}", "index": i, "type": str})
    return fmt
