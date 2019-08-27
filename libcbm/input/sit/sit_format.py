import numpy as np


classifier_keyword = "_CLASSIFIER"


def get_classifier_format(n_columns):
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
        {"name": "id", "index": 0},
        {"name": "name", "index": 1},
        {"name": "description", "index": 2},
    ]

    if n_columns < 3:
        raise ValueError(
            "specified number of columns invalid.  Expected at least 3.")
    elif n_columns > 4:
        classifier_format.extend([{
            "name": "aggregate_value_{}".format(i-2), "index": i
        } for i in range(3, n_columns)])
    return classifier_format


def get_disturbance_type_format(n_columns):
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
        {"name": "id", "index": 0},
        {"name": "name", "index": 1},
    ]
    if n_columns < 2:
        raise ValueError(
            "specified number of columns invalid.  Expected at least 2.")
    elif n_columns == 3:
        disturbance_type_format.append({"name": "description", "index": 2})
    elif n_columns > 3:
        raise ValueError(
            "specified number of columns invalid.  Expected at most 3.")
    return disturbance_type_format


def get_age_class_format():
    """Gets a list of dictionaries describing the CBM SIT age class columns

    Returns:
        list: a list of dictionaries that describe the  CBM SIT age class
            columns
    """
    return [
        {"name": "id", "index": 0},
        {"name": "size", "index": 1, "type": np.int},
    ]


def get_yield_format(classifier_names, n_columns):
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
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]
    leading_species_col = [{
        "name": "leading_species", "index": n_classifiers}]
    vol_index = n_classifiers + 1
    if n_columns < vol_index + 1:
        # in SIT vol_index is the 0th age volume, so at least 2 volumes need
        # to be here for anything to happen in CBM
        raise ValueError(
            "at least {0} columns are required".format(vol_index + 1))
    volumes = [
        {"name": "v{}".format(i), "index": i, "min_value": 0, "type": np.float}
        for i in range(vol_index, n_columns)]

    return classifier_values + leading_species_col + volumes


def get_age_eligibility_columns(base_index):
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
        {"name": "min_softwood", "index": base_index + 1},
        {"name": "max_softwood", "index": base_index + 2},
        {"name": "min_hardwood", "index": base_index + 3},
        {"name": "max_hardwood", "index": base_index + 4}
    ]


def get_transition_rules_format(classifier_names, n_columns):
    """Generate a list of dictionaries describing each column in the SIT
    format transition rules.  The format is dynamic and changes based on the
    number of classifiers and whether or not a spatial identifier is
    specified.

    Args:
        classifier_names (int): a list of the names of classifiers
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
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]
    age_eligibility = get_age_eligibility_columns(n_classifiers)
    disturbance_type = [{
        "name": "disturbance_type",
        "index": n_classifiers+5}]

    regeneration_delay_index = 2 * n_classifiers + len(age_eligibility) + 1
    post_transition = [
        {"name": "regeneration_delay", "index": regeneration_delay_index,
            "min_value": 0, "type": np.float},
        {"name": "reset_age", "index": regeneration_delay_index + 1,
            "min_value": -1, "type": np.int},
        {"name": "percent", "index": regeneration_delay_index + 2,
            "min_value": 0, "max_value": 100, "type": np.int},
    ]
    spatial_reference = [
        {"name": "spatial_reference", "index": regeneration_delay_index + 3,
            "type": np.int}
    ]
    classifier_set_dest = [
        {"name": c, "index": i + n_classifiers + len(age_eligibility) + 1}
        for i, c in enumerate(classifier_names)]
    result = []
    result.extend(classifier_set_src)  # source classifier set
    result.extend(age_eligibility)
    result.extend(disturbance_type)
    result.extend(classifier_set_dest)  # destination classifier set
    result.extend(post_transition)
    if n_columns < len(result):
        raise ValueError(
            "specified number of columns invalid.  Expected at least "
            "{}".format(len(result)))
    if n_columns == len(result):
        return result
    elif n_columns == len(result) + 1:
        result.extend(spatial_reference)
        return result
    else:
        raise ValueError(
            "incorrect number of columns in transition rules. "
            "Expected at most {}".format(len(result) + 1))


def get_inventory_format(classifier_names, n_columns):
    """Gets a description of the SIT inventory columns as a list of
    dictionaries

    Args:
        classifier_names (int): a list of the names of classifiers
            n_columns (int): the number of columns in inventory data.  This
            is required because the format has a varying number of optional
            columns.

    Raises:
        ValueError: The number of columns was incorrect

    Returns:
        list: a list of dictionaries describing the SIT inventory columns
    """
    n_classifiers = len(classifier_names)

    classifier_set = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]

    inventory = [
        {"name": "using_age_class", "index": n_classifiers},
        {"name": "age", "index": n_classifiers + 1, "min_value": 0},
        {"name": "area", "index": n_classifiers + 2, "min_value": 0},
        {"name": "delay", "index": n_classifiers + 3, "min_value": 0},
        {"name": "land_class_id", "index": n_classifiers + 4}]

    if n_columns > n_classifiers + 6:
        inventory.extend([
            {"name": "historical_disturbance_type",
             "index": n_classifiers + 5},
            {"name": "last_pass_disturbance_type",
             "index": n_classifiers + 6}])
    if n_columns == n_classifiers + 6:
        raise ValueError(
            "Invalid number of columns: both historical and last pass "
            "disturbance types must be defined.")
    if n_columns < n_classifiers + 5:
        raise ValueError(
            f"With {n_classifiers} classifiers, SIT inventory should have "
            f"at least {n_classifiers + 5} columns.")
    if n_columns == n_classifiers + 8:
        inventory.append({
            "name": "spatial_reference", "index": n_classifiers + 7,
            "type": np.int})
    if n_columns > n_classifiers + 8:
        raise ValueError(
            f"With {n_classifiers} classifiers, SIT inventory should have "
            f"at most {n_classifiers + 8} columns.")

    return classifier_set + inventory


def get_disturbance_eligibility_columns(index):
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
        {"name": "MinYearsSinceDist", "index": index + 0, "type": np.float},
        {"name": "MaxYearsSinceDist", "index": index + 1, "type": np.float},
        {"name": "LastDistTypeID", "index": index + 2, "type": np.float},
        {"name": "MinTotBiomassC", "index": index + 3, "type": np.float},
        {"name": "MaxTotBiomassC", "index": index + 4, "type": np.float},
        {"name": "MinSWMerchBiomassC", "index": index + 5, "type": np.float},
        {"name": "MaxSWMerchBiomassC", "index": index + 6, "type": np.float},
        {"name": "MinHWMerchBiomassC", "index": index + 7, "type": np.float},
        {"name": "MaxHWMerchBiomassC", "index": index + 8, "type": np.float},
        {"name": "MinTotalStemSnagC", "index": index + 9, "type": np.float},
        {"name": "MaxTotalStemSnagC", "index": index + 10, "type": np.float},
        {"name": "MinSWStemSnagC", "index": index + 11, "type": np.float},
        {"name": "MaxSWStemSnagC", "index": index + 12, "type": np.float},
        {"name": "MinHWStemSnagC", "index": index + 13, "type": np.float},
        {"name": "MaxHWStemSnagC", "index": index + 14, "type": np.float},
        {"name": "MinTotalStemSnagMerchC", "index": index + 15,
            "type": np.float},
        {"name": "MaxTotalStemSnagMerchC", "index": index + 16,
            "type": np.float},
        {"name": "MinSWMerchStemSnagC", "index": index + 17, "type": np.float},
        {"name": "MaxSWMerchStemSnagC", "index": index + 18, "type": np.float},
        {"name": "MinHWMerchStemSnagC", "index": index + 19, "type": np.float},
        {"name": "MaxHWMerchStemSnagC", "index": index + 20, "type": np.float}
    ]


def get_disturbance_event_format(classifier_names, n_columns):

    n_classifiers = len(classifier_names)

    classifier_set = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]

    disturbance_age_eligibility = get_age_eligibility_columns(n_classifiers)

    n_age_fields = len(disturbance_age_eligibility)
    disturbance_eligibility = get_disturbance_eligibility_columns(
        n_classifiers + n_age_fields)
    n_eligibility_fields = len(disturbance_eligibility)
    index = n_classifiers + n_age_fields + n_eligibility_fields
    event_target = [
        {"name": "efficiency", "index": index, "type": np.float,
            "min_value": 0, "max_value": 1},
        {"name": "sort_type", "index": index + 1},
        {"name": "target_type", "index": index + 2},
        {"name": "target", "index": index + 3, "type": np.float},
        {"name": "disturbance_type", "index": index + 4},
        {"name": "disturbance_year", "index": index + 5, "type": np.int},
    ]
    if n_columns < index + 6:
        raise ValueError(
            "specified number of columns invalid.  Expected at least "
            "{}".format(len(index + 6)))
    if n_columns == index + 7:
        event_target.append(
            {"name": "spatial", "index": index + 6}
        )
    if n_columns > index + 7:
        raise ValueError(
            "specified number of columns invalid.  Expected at most "
            "{}".format(len(index + 7)))
    return classifier_set + disturbance_age_eligibility + \
        disturbance_eligibility + event_target
