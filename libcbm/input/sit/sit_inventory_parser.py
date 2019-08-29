import pandas as pd
import numpy as np
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def parse_inventory(inventory_table, classifiers, classifier_values,
                    disturbance_types, age_classes, land_classes):
    inventory_format = sit_format.get_inventory_format(
            classifiers.name,
            len(inventory_table.columns))

    inventory = sit_parser.unpack_table(
        inventory_table, inventory_format, "inventory")

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

    undefined_land_classes = set()
    inventory.land_class = inventory.land_class.apply(
        get_map_land_class_func(land_classes, undefined_land_classes.add))
    if len(undefined_land_classes) > 0:
        raise ValueError(
            "inventory land_class column contains undefined land class ids: "
            f"{undefined_land_classes}")

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

    # for rows where using_age_class is false, a type of integer and min value
    # of 0 is enforced
    age_column_format = [
        x for x in inventory_format if x["name"] == "age"][0].copy()
    age_column_format["type"] = np.int32
    age_column_format["min_value"] = 0

    sit_parser.unpack_column(
        inventory.loc[~inventory.using_age_class],
        age_column_format, "inventory")

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
    for row in age_classes.itertuples():

        age_range = range(row.start_year, row.start_year + row.class_size) \
            if row.class_size > 0 else [0]

        expanded_age_classes = expanded_age_classes.append(
            pd.DataFrame({
                "name": row.name,
                "age": age_range,
                "class_size": row.class_size}))

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
    age_class_merge.loc[age_class_merge.class_size > 0, "area"] \
        /= age_class_merge.class_size[age_class_merge.class_size > 0]

    age_class_merge = age_class_merge.rename(columns={"age_x": "age"})
    age_class_merge = age_class_merge.drop(
        columns=["age_y", "class_size", "name"])
    result = non_using_age_class_rows.append(age_class_merge) \
        .reset_index(drop=True)

    return result


def get_map_land_class_func(land_classes, on_error):
    """Returns a function for mapping land class to land class id.

    Args:
        land_classes (dict): a dictionary of landclass id (int, key)
            to land class value (str, value)
        on_error (func): a function of a single parameter, if the
            specified id is not found call this function with that id

    Returns:
        str, or None: the mapped value if it exists or None
    """
    def map_land_class(id):
        """function for mapping land class to land class id.

        Args:
            id (int): land class id

        Returns:
            str, or None: the mapped value if it exists or None
        """
        try:
            return land_classes[id]
        except KeyError:
            on_error(id)
            return None
    return map_land_class