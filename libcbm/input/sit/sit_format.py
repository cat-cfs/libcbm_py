import numpy as np


classifier_keyword = "_CLASSIFIER"


def get_classifier_format():
    return [
        {"name": "id", "index": 0},
        {"name": "name", "index": 1},
        {"name": "description", "index": 2},
        {"name": "classifier_aggregates", "column_range": [3, -1]}
    ]


def get_disturbance_type_format():
    return [
        {"name": "id", "index": 0},
        {"name": "name", "index": 1},
        {"name": "description", "index": 2, "required": False},
    ]


def get_age_class_format():
    return [
        {"name": "id", "index": 0},
        {"name": "size", "index": 1, "type": np.int},
    ]


def get_yield_format(classifier_names):
    n_classifiers = len(classifier_names)
    classifier_values = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]
    leading_species_col = [{
        "name": "leading_species", "index": n_classifiers}]
    volumes = [{
        "name": "volumes", "column_range": [n_classifiers+1, -1],
        "min_value": 0, "type": np.float}]
    return classifier_values + leading_species_col + volumes


def get_transition_rules_format(classifier_names, n_columns):
    """Generate a list of dictionaries describing each column in the SIT
    format transition rules.  The format is dynamic and changes based on the
    number of classifiers and whether or not a spatial identifier is
    specified.

    Args:
        classifier_names (int): a list of the names of classifiers
        n_columns (int): the number of columns in transition rules data.
        This is used to detect whether or not a spatial identifier is included
        in the data.

    Raises:
        ValueError: n_columns did not match the possible number of columns
            in the SIT transitions format.

    Returns:
        list: a list of dictionaries describing the SIT transition rule columns
    """
    n_classifiers = len(classifier_names)
    classifier_set = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]
    age_eligibility = [
        {"name": "using_age_class", "index": n_classifiers},
        {"name": "min_softwood", "index": n_classifiers + 1},
        {"name": "max_softwood", "index": n_classifiers + 2},
        {"name": "min_hardwood", "index": n_classifiers + 3},
        {"name": "max_hardwood", "index": n_classifiers + 4}
    ]
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
    result = []
    result.extend(classifier_set)  # source classifier set
    result.extend(age_eligibility)
    result.extend(disturbance_type)
    result.extend(classifier_set)  # destination classifier set
    result.extend(post_transition)
    if n_columns == len(result):
        return result
    elif n_columns == len(result) + 1:
        result.extend(spatial_reference)
        return result
    else:
        raise ValueError("incorrect number of columns in transition rules")


def get_inventory_format(classifier_names, n_columns):
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
        if n_columns == 6:
            raise ValueError(
                "Invalid number of columns: both historical and last pass "
                "disturbance types must be defined.")
        inventory.extend([
            {"name": "historical_disturbance_type",
             "index": n_classifiers + 5},
            {"name": "last_pass_disturbance_type",
             "index": n_classifiers + 6}])

    if n_columns == n_classifiers + 8:
        inventory.append({
            "name": "spatial_reference", "index": n_classifiers + 7,
            "type": np.int})

    return classifier_set + inventory


def get_disturbance_event_format(classifier_names, n_columns):

    n_classifiers = len(classifier_names)

    classifier_set = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]

    disturbance_age_eligibility = [
        {"name": "using_age_class", "index": n_classifiers},
        {"name": "min_softwood", "index": n_classifiers + 1},
        {"name": "max_softwood", "index": n_classifiers + 2},
        {"name": "min_hardwood", "index": n_classifiers + 3},
        {"name": "max_hardwood", "index": n_classifiers + 4}
    ]

    n_age_fields = len(disturbance_age_eligibility)
    index = n_classifiers + n_age_fields
    disturbance_eligibility = [
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
        {"name": "MinTotalStemSnagMerchC", "index": index + 15, "type": np.float},
        {"name": "MaxTotalStemSnagMerchC", "index": index + 16, "type": np.float},
        {"name": "MinSWMerchStemSnagC", "index": index + 17, "type": np.float},
        {"name": "MaxSWMerchStemSnagC", "index": index + 18, "type": np.float},
        {"name": "MinHWMerchStemSnagC", "index": index + 19, "type": np.float},
        {"name": "MaxHWMerchStemSnagC", "index": index + 20, "type": np.float}
    ]

    event_target = []