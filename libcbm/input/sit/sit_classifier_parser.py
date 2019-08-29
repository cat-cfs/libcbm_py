import pandas as pd
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def get_classifier_keyword():
    """gets the _CLASSIFIER keyword using the SIT_Classifiers format.

    Returns:
        str: _CLASSIFIER
    """
    return "_CLASSIFIER"


def parse_classifiers(classifiers_table):
    """parse SIT_Classifiers formatted data

    Args:
        classifiers_table (pandas.DataFrame): a dataFrame in sit classifiers
            format.

    Raises:
        ValueError: duplicated names detected, or other validation error
            occurred

    Returns:
        tuple:

            - classifiers - a validated table of classifiers
            - classifier_values - a validated table of classifier values
            - aggregate_values - a dictionary describing aggregate values

    """
    classifiers_format = sit_format.get_classifier_format(
        len(classifiers_table.columns))
    unpacked = sit_parser.unpack_table(
        classifiers_table, classifiers_format, "classifiers"
    )
    classifiers = unpacked \
        .loc[unpacked["name"] == get_classifier_keyword()]
    classifiers = pd.DataFrame(
        data={
            "id": classifiers.id,
            # for classifiers, the 3rd column is used for the name
            "name": classifiers.description},
        columns=["id", "name"])
    duplicate_classifiers = classifiers.groupby("name").size()
    duplicated_classifier_names = list(
        duplicate_classifiers[duplicate_classifiers > 1].index)
    if len(duplicated_classifier_names) > 0:
        raise ValueError(
            "The following classifier names appear more than one time:"
            f"{duplicated_classifier_names}")
    # filter out rows that have the _CLASSIFIER keyword and also
    # any rows that have a value on the 3rd or greater column.
    # This is the set of classifier values.
    classifier_values = unpacked \
        .loc[pd.isnull(unpacked.iloc[:, 3:]).all(axis=1) &
             (unpacked["name"] != get_classifier_keyword())]

    classifier_values = pd.DataFrame({
        "classifier_id": classifier_values.id,
        "name": classifier_values.name,
        "description": classifier_values.description
    })

    duplicate_classifier_values = classifier_values.groupby(
        ["classifier_id", "name"]).size()
    duplicate_classifier_values = [
        {"classifier_id": x[0], "classifier_value": x[1]}
        for x in list(
            duplicate_classifier_values[
                duplicate_classifier_values > 1].index)]
    if len(duplicate_classifier_values) > 0:
        raise ValueError(
            "The following classifier values are duplicated for the specified "
            f"classifier ids: {duplicate_classifier_values}")

    aggregate_values = []
    classifier_aggregates = unpacked.loc[
        ~pd.isnull(unpacked.iloc[:, 3:]).all(axis=1)]
    for i in range(0, classifier_aggregates.shape[0]):

        agg_values = classifier_aggregates.iloc[i, 3:]
        agg_values = agg_values[~pd.isnull(agg_values)]
        aggregate_values.append({
            "classifier_id": classifier_aggregates.iloc[i, :]["id"],
            "name": classifier_aggregates.iloc[i, :]["name"],
            "description": classifier_aggregates.iloc[i, :]["description"],
            "classifier_values": list(agg_values[:])
        })

    unique_agg_set = set()
    unique_agg_value_set = set()
    for agg in aggregate_values:
        classifier_id = agg["classifier_id"]
        name = agg["name"]
        agg_values = agg["classifier_values"]
        if len(agg_values) > len(set(agg_values)):
            raise ValueError(
                "duplicate classifier values detected in aggregate with "
                f"classifier_id: {classifier_id}, name {name}")
        for classifier_value in agg_values:
            unique_agg_value_set.add((classifier_id, classifier_value))
        if (classifier_id, name) in unique_agg_set:
            raise ValueError(
                "duplicate classifier aggregate detected: "
                f"classifier_id: {classifier_id}, name {name}")
        else:
            unique_agg_set.add((classifier_id, name))

    for classifier_id in classifier_values.classifier_id.unique():
        classifier_id_values_set = set(
            classifier_values[
                classifier_values.classifier_id == classifier_id].name)
        aggregate_values_set = set(
            [x[1] for x in unique_agg_value_set if x[0] == classifier_id])
        if not aggregate_values_set.issubset(classifier_id_values_set):
            missing_aggregate_values = aggregate_values_set.difference(
                classifier_id_values_set)
            raise ValueError(
                "The following aggregate values that are not defined as "
                f"classifier values in the classifier with id {classifier_id} "
                f"were found: {missing_aggregate_values}.")
    return classifiers, classifier_values, aggregate_values
