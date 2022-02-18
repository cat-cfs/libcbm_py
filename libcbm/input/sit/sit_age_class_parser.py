# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def generate_sit_age_classes(age_interval: int, max_age: int) -> pd.DataFrame:
    """generate a valid SIT_ageclass input table.  This is a helper method
    to create input for :py:func:`parse_age_classes`

    Args:
        age_interval (int): the number of years between age classes
        max_age (int): the maximum age

    Returns:
        pandas.DataFrame: a table of valid SIT_AgeClasses based on the
            parameters

    Examples:
        >>> generate_sit_age_classes(2, 10)
           0  1
        0  0  0
        1  1  2
        2  2  2
        3  3  2
        4  4  2
        5  5  2
    """

    if age_interval <= 0:
        raise ValueError("age_interval must be > 0")
    if max_age <= 0:
        raise ValueError("max_age must be > 0")

    length = len(range(1, max_age, age_interval))
    data = [(0, 0)]
    data += [(i + 1, age_interval) for i in range(length)]
    return pd.DataFrame(data)


def parse(age_class_table: pd.DataFrame) -> pd.DataFrame:
    """Parse the sit age class table format into a table of age classes with
    fields:

        - name
        - class_size
        - start_year
        - end_year

    Args:
        age_class_table (pandas.DataFrame): a dataframe

    Raises:
        ValueError: the first, and only the first row must have a 0 value
        ValueError: duplicate values in the first column of the specified
            table were detected

    Example:

        Input:

            ======  ====
             0       1
            ======  ====
            age_0    0
            age_1    10
            age_2    10
            age_3    10
            age_4    10
            age_5    10
            age_6    10
            age_7    10
            age_8    10
            age_9    10
            ======  ====

        Output:

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


    Returns:
        pandas.DataFrame: a dataframe describing the age classes.
    """
    table = sit_parser.unpack_table(
        age_class_table, sit_format.get_age_class_format(), "age classes"
    )

    result = []
    for i, row in enumerate(table.itertuples()):
        size = row.class_size
        if i == 0:
            if size != 0:
                raise ValueError("First age class row expected to have 0 size")
            result.append(
                {
                    "name": row.id,
                    "class_size": 0,
                    "start_year": 0,
                    "end_year": 0,
                }
            )
        else:
            start_year = result[-1]["end_year"] + 1
            if size == 0:
                raise ValueError(
                    "All age class rows other than the"
                    "first one must have size > 0"
                )
            result.append(
                {
                    "name": row.id,
                    "class_size": row.class_size,
                    "start_year": start_year,
                    "end_year": start_year + row.class_size - 1,
                }
            )

    age_classes = pd.DataFrame(
        result, columns=["name", "class_size", "start_year", "end_year"]
    )

    duplicates = age_classes.groupby("name").size()
    duplicates = list(duplicates[duplicates > 1].index)
    if len(duplicates) > 0:
        raise ValueError(
            f"duplicate names detected in age classes {duplicates}"
        )
    return age_classes
