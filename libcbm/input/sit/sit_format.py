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
        {"name": "size", "index": 1, "type": "int"},
    ]


def get_yield_format(classifier_names):
    n_classifiers = len(classifier_names)
    classifier_values = [
        {"name": c, "index": i} for i, c in enumerate(classifier_names)]
    leading_species_col = [{
        "name": "leading_species", "index": n_classifiers}]
    volumes = [{
        "name": "volumes", "column_range": [n_classifiers+1, -1],
        "min_value": 0, "type": "double"}]
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
            "min_value": 0, "type": "double"},
        {"name": "reset_age", "index": regeneration_delay_index + 1,
            "min_value": -1, "type": "int"},
        {"name": "percent", "index": regeneration_delay_index + 2,
            "min_value": 0, "max_value": 100, "type": "int"},
    ]
    spatial_reference = [
        {"name": "spatial_reference", "index": regeneration_delay_index + 3,
            "type": "int"}
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
