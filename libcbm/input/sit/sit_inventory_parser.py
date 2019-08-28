import pandas as pd
import numpy as np
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def parse_inventory(inventory_table, classifiers, classifier_values,
                    disturbance_types, age_classes):
    inventory = sit_parser.unpack_table(
        inventory_table,
        sit_format.get_inventory_format(
            classifiers.name,
            len(inventory_table.columns)),
        "inventory")

    # validate the classifier values in the inventory table
    for row in classifiers.itertuples():
        a = inventory[row.name].unique()
        b = classifier_values[
            classifier_values["classifier_id"] == row.id]["name"].unique()
        diff = np.setdiff1d(a, b)
        if len(diff) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff}")

    # if the historical/last pass disturbances are specified substitute them
    # according to the specified disturbance type parameters
    if "historical_disturbance_type" in inventory:
        # first of all, validate
        undefined_historic = np.setdiff1d(
            inventory.historical_disturbance_type.unique(),
            disturbance_types.id.unique())

        undefined_lastpass = np.setdiff1d(
            inventory.last_pass_disturbance_type.unique(),
            disturbance_types.id.unique())
        if len(undefined_historic) > 0:
            raise ValueError(
                "Undefined disturbance type ids (as defined in sit "
                f"disturbance types) detected: {undefined_historic}"
            )
        if len(undefined_lastpass) > 0:
            raise ValueError(
                "Undefined disturbance type ids (as defined in sit "
                f"disturbance types) detected: {undefined_lastpass}"
            )

        historic_join = inventory.merge(
            disturbance_types, left_on="historical_disturbance_type",
            right_on="id")
        last_pass_join = inventory.merge(
            disturbance_types, left_on="historical_disturbance_type",
            right_on="id")
        inventory.historical_disturbance_type = historic_join.name
        inventory.last_pass_disturbance_type = last_pass_join.name

    inventory.using_age_class = inventory.using_age_class.map(
        sit_parser.get_parse_bool_func("inventory", "using_age_class"))

    if inventory.using_age_class.any():
        inventory = expand_age_class_inventory(inventory, age_classes)

    return inventory


def expand_age_class_inventory(inventory, age_classes):
    expanded_age_classes = pd.DataFrame()

    undefined_age_class_name = np.setdiff1d(
        inventory.loc[
            inventory.using_age_class].age.astype(str).unique(),
        age_classes.name.unique())
    if len(undefined_age_class_name) > 0:
        raise ValueError(
            "Undefined age class ids (as defined in sit "
            f"age classes) detected: {undefined_age_class_name}"
        )
    for i, row in enumerate(age_classes.itertuples()):

        age_range = range(row.start_year, row.start_year + row.size) \
            if row.size > 0 else [0]

        expanded_age_classes = expanded_age_classes.append(
            pd.DataFrame({
                "name": row.name,
                "age": age_range,
                "size": row.size}))

    non_using_age_class_rows = inventory.loc[~inventory.using_age_class]
    using_age_class_rows = inventory.loc[inventory.using_age_class].copy()

    if "spatial_reference" in using_age_class_rows:
        if (using_age_class_rows.spatial_reference >= 0).any():
            raise ValueError(
                "using_age_class=true and spatial reference may not be "
                "used together")
    using_age_class_rows.age = using_age_class_rows.age.astype(np.str)

    age_class_merge = using_age_class_rows.merge(
        expanded_age_classes, left_on="age", right_on="name")

    age_class_merge.age_x = (age_class_merge.age_y)
    age_class_merge.loc[age_class_merge["size"] > 0, "area"] \
        /= age_class_merge["size"][age_class_merge["size"] > 0]

    age_class_merge = age_class_merge.rename(columns={"age_x": "age"})
    age_class_merge = age_class_merge.drop(
        columns=["age_y", "size", "name"])
    result = non_using_age_class_rows.append(age_class_merge) \
        .reset_index(drop=True)

    return result
