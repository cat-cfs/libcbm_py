import pandas as pd
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def get_classifier_keyword():
    return "_CLASSIFIER"


def parse_classifiers(classifiers_table):

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
    return classifiers, classifier_values, aggregate_values
