# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def parse(inventory_table, classifiers, classifier_values,
          disturbance_types, age_classes):
    """Parses and validates SIT formatted inventory data.  The inventory_table
    parameter is the primary data, and the other args act as validation
    metadata.

    Args:
        inventory_table (pandas.DataFrame): SIT formatted inventory
        classifiers (pandas.DataFrame): table of classifier as returned by the
            function:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): table of classifier values as
            returned by the function:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        disturbance_types (pandas.DataFrame): table of disturbance types as
            returned by the function:
            :py:func:`libcbm.input.sit.sit_disturbance_type_parser.parse`
        age_classes (pandas.DataFrame): table of disturbance types as
            returned by the function:
            :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Raises:
        ValueError: Undefined classifier values detected in inventory table
        ValueError: Undefined disturbance types detected in inventory table

    Example:

        Input:

            SIT_Inventory:

                ===  ===  ======  =======  ===  ===  ===  =====  =====  ===
                0    1    2       3        4    5    6    7       8      9
                ===  ===  ======  =======  ===  ===  ===  =====  =====  ===
                b    a    True    age_2    1    1    1    dist1  dist2  -1
                a    a    False   100      1    0    0    dist2  dist1   0
                a    a    -1      4        1    0    0    dist1  dist1  -1
                ===  ===  ======  =======  ===  ===  ===  =====  =====  ===

            classifiers parameter:

                ===  ===========
                id   name
                ===  ===========
                1    classifier1
                2    classifier2
                ===  ===========

            classifier_values parameter:

                ==============  =====  ============
                classifier_id   name   description
                ==============  =====  ============
                 1               a      a
                 1               b      b
                 2               a      a
                ==============  =====  ============

            disturbance_types parameter:

                ======  =========
                id         name
                ======  =========
                dist1    fire
                dist2    clearcut
                dist3    clearcut
                ======  =========

            age_classes parameter:

                ======  ===========  ===========  =========
                name    class_size   start_year   end_year
                ======  ===========  ===========  =========
                age_0    0              0           0
                age_1    10             1           10
                age_2    10             11          20
                age_3    10             21          30
                age_4    10             31          40
                age_5    10             41          50
                age_6    10             51          60
                age_7    10             61          70
                age_8    10             71          80
                age_9    10             81          90
                ======  ===========  ===========  =========

            land_classes parameter::

                land_classes = {0: "lc_1", 1: "lc_2"}

        Output: (abbreviated column names)

            ==  ===    =====  ====  =====  =====  ==========  =========  =====
            c1  c2     age    area  delay   lc    hist_dist   last_dist  s_ref
            ==  ===    =====  ====  =====  =====  ==========  =========  =====
            a    a      100   1.0    0      lc_1    fire       fire        0
            a    a      4     1.0    0      lc_1    clearcut   clearcut   -1
            b    a      11    0.1    1      lc_2    fire       fire       -1
            b    a      12    0.1    1      lc_2    fire       fire       -1
            b    a      13    0.1    1      lc_2    fire       fire       -1
            b    a      14    0.1    1      lc_2    fire       fire       -1
            b    a      15    0.1    1      lc_2    fire       fire       -1
            b    a      16    0.1    1      lc_2    fire       fire       -1
            b    a      17    0.1    1      lc_2    fire       fire       -1
            b    a      18    0.1    1      lc_2    fire       fire       -1
            b    a      19    0.1    1      lc_2    fire       fire       -1
            b    a      20    0.1    1      lc_2    fire       fire       -1
            ==  ===    =====  ====  =====  =====  ==========  =========  =====

            The actual output column names for this example are:

                - classifier1
                - classifier2
                - age
                - area
                - delay
                - land_class
                - historical_disturbance_type
                - last_pass_disturbance_type
                - spatial_reference

    Returns:
        pandas.DataFrame: validated inventory
    """
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

    inventory = inventory.drop(columns=["using_age_class"])
    inventory = inventory.reset_index(drop=True)

    if "spatial_reference" in inventory:
        if inventory.spatial_reference[
                inventory.spatial_reference > 0].duplicated().any():
            raise ValueError(
                "duplicate value detected in spatial_reference column")
    return inventory


def expand_age_class_inventory(inventory, age_classes):
    """Support for the SIT age class inventory feature.  For rows with
    inventory.using_age_class = True, the inventory.age column represents an
    identifier defined in the passed age_classes table.  The inventory record
    is divided into one record per year in the associated age class with the
    full range of ages.

    Args:
        inventory (pandas.DataFrame): [description]
        age_classes (pandas.DataFrame): table of disturbance types as
            returned by the function:
            :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Raises:
        ValueError: Undefined age class ids found in inventory
        ValueError: Age class inventory mixed with spatial identifier

    Returns:
        pandas.DataFrame: the age class expanded inventory
    """
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
    def map_land_class(land_class_id):
        """function for mapping land class to land class id.

        Args:
            land_class_id (int): land class id

        Returns:
            str, or None: the mapped value if it exists or None
        """
        try:
            return land_classes[land_class_id]
        except KeyError:
            on_error(id)
            return None
    return map_land_class
