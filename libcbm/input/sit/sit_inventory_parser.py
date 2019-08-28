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
    if "historical_disturbance_type" in {inventory.columns}:
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

    return inventory
