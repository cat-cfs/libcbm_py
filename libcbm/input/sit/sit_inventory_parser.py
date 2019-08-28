import pandas as pd
import numpy as np
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def parse_inventory(inventory_table, classifiers, classifier_values):
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

    return inventory
