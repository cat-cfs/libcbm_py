# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np


def unpack_column(table, column_description, table_name):
    """Validates a column in a pandas DataFrame

    Args:
        table ([type]): A table containing the column at the index specified
            by column_description
        column_description (dict): A dictionary with the following supported
            keys:

             - name: the name of the column, which is assigned to the column
                     label to the result table returned by this function
             - index: the zero based index of the column in the table's
                      ordered columns
             - type: (optional) the column will be converted (if necessary to
                     this type) If the conversion is not possible for a value
                     at any row, an error is raised.
             - min_value: (optional) inclusive minimum value constraint.
             - max_value: (optional) inclusive maximum value constraint.

        table_name (str): the name of the table being processed, purely
            for error feedback when an error occurs.

    Raises:
        ValueError: the values in the column were not convertable to the
            specified column description type
        ValueError: a min_value or max_value was specified without specifying
            type in column description
        ValueError: the min_value or max_value constraint was violated by the
            value in the column.

    Returns:
        pandas.DataFrame: the resulting table
    """
    data = table.iloc[:, column_description["index"]]
    col_name = column_description["name"]
    if "type" in column_description:
        try:
            data = data.astype(column_description['type'], skipna=True)
        except ValueError:
            raise ValueError(
                f"{table_name} table, column: '{col_name}' contains values "
                f"that cannot be converted to: '{column_description['type']}'")
    if "min_value" in column_description:
        if "type" not in column_description:
            raise ValueError("type required with min_value")
        min_value = column_description["min_value"]
        if len(data[data < min_value]):
            raise ValueError(
                f"{table_name} table, column: '{col_name}' contains values "
                f"less than the minimum allowed value: {min_value}")
    if "max_value" in column_description:
        if "type" not in column_description:
            raise ValueError("type required with max_value")
        max_value = column_description["max_value"]
        if len(data[data > max_value]):
            raise ValueError(
                f"{table_name} table, column: '{col_name}' contains values "
                f"greater than the maximum allowed value: {max_value}")
    return data


def _list_duplicates(seq):
    """
    get the list of duplicate values in the specified sequence
    https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them

    Args:
        seq (iterable): a sequence of comparable values

    Returns:
        list: the list of values that appear at least 2 times in the input
    """
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


def unpack_table(table, column_descriptions, table_name):
    """Validates and assigns column names to a column-ordered table using
    the specified list of column descriptions. Any existing column labels
    on the specified table are ignored.

    Args:
        table (pandas.DataFrame): a column ordered table to validate
        column_descriptions (list): a list of dictionaries with describing the
            columns. See :py:func:`unpack_column` for how this is used
        table_name (str): the name of the table being processed, purely
            for error feedback when an error occurs.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    cols = [x["name"] for x in column_descriptions]
    duplicates = _list_duplicates(cols)
    if duplicates:
        # this could potentially happen if a classifier is named the same
        # thing as another column
        raise ValueError(f"duplicate column names detected: {duplicates}")
    data = {
        x["name"]: unpack_column(table, x, table_name)
        for x in column_descriptions}
    return pd.DataFrame(columns=cols, data=data)


def _try_get_int(s):
    """
    Checks if the specified value is an integer, and returns a result

    Args:
        s (any): a value to test

    Returns:
        tuple: (int(s), True) if the value can be converted to an integer,
            and otherwise (None, False)
    """
    try:
        i = int(s)
        return i, True
    except ValueError:
        return None, False


def get_parse_bool_func(table_name, colname):
    """gets a boolean-like value to boolean parse function according to the
    SIT specification.  The parameters are used to create a friendly error
    message when a parse failure occurs.

    Args:
        table_name (str): Table name to be used in failure error message
        colname (str): Column name to be used in failure error message

    Returns:
        func: a boolean-like value to bool parse function
    """
    def parse_bool(x):
        """Converts the specified value to a boolean according to SIT
        specification, or raises an error.

        Args:
            x (varies): a value to convert to boolean

        Raises:
            ValueError: The specified value was not convertable to boolean

        Returns:
            boolean: The converted value
        """
        if isinstance(x, bool):
            return x
        elif isinstance(x, int):
            # the sit format treats negatives as False for boolean fields
            return x > 0
        else:
            str_x = str(x).lower()
            int_x, success = _try_get_int(str_x)
            if success:
                return int_x > 0
            if str_x in ["true", "t", "y"]:
                return True
            elif str_x in ["false", "f", "n"]:
                return False
            else:
                raise ValueError(
                    f"{table_name}: cannot parse value: '{x}' in "
                    f"column: '{colname}' as a boolean")

    return parse_bool


def substitute_using_age_class_rows(rows, parse_bool_func, age_classes):
    """Substitute age class criteria values that appear in SIT transition
    rules or disturbance events data into age values.

    Checks that min softwood age equals min hardwood age and max softwood
    age equals max hardwood age since CBM does not carry separate HW/SW ages.

    Args:
        rows (pandas.DataFrame): sit data containing columns that describe age
            eligibility:

            - using_age_class
            - min_softwood_age
            - min_hardwood_age
            - max_softwood_age
            - max_hardwood_age

        parse_bool_func (func): a function that maps boolean-like values to
            boolean.  Passed to the pandas.Series.map function for the
            using_age_class column.
        age_classes (pandas.DataFrame): [description]

    Raises:
        ValueError: values found in the age eligibility columns are not
            defined identifiers in the specified age classes table.
        ValueError: hardwood and softwood age criteria were not identical.

    Returns:
        pandas.DataFrame: the input table with age values criteria substituted
            for age class criteria.
    """

    rows.using_age_class = \
        rows.using_age_class.map(parse_bool_func)
    non_using_age_class_rows = rows.loc[
        ~rows.using_age_class]
    using_age_class_rows = rows.loc[
        rows.using_age_class].copy()

    for age_class_criteria_col in [
          "min_softwood_age", "min_hardwood_age",
          "max_softwood_age", "max_hardwood_age"]:
        valid_age_classes = np.concatenate(
            [age_classes.name.unique(), np.array(["-1"])])
        age_class_ids = using_age_class_rows[
            age_class_criteria_col].astype(np.str).unique()
        undefined_age_classes = np.setdiff1d(
            age_class_ids,
            valid_age_classes)
        if len(undefined_age_classes) > 0:
            raise ValueError(
                f"In column {age_class_criteria_col}, the following age class "
                f"identifiers: {undefined_age_classes} are not defined in SIT "
                "age classes.")

    age_class_start_year_map = {
        x.name: x.start_year for x in age_classes.itertuples()}
    age_class_end_year_map = {
        x.name: x.end_year for x in age_classes.itertuples()}
    using_age_class_rows.min_softwood_age = using_age_class_rows \
        .min_softwood_age.astype(np.str).map(age_class_start_year_map)
    using_age_class_rows.min_hardwood_age = using_age_class_rows \
        .min_hardwood_age.astype(np.str).map(age_class_start_year_map)
    using_age_class_rows.max_softwood_age = using_age_class_rows \
        .max_softwood_age.astype(np.str).map(age_class_end_year_map)
    using_age_class_rows.max_hardwood_age = using_age_class_rows \
        .max_hardwood_age.astype(np.str).map(age_class_end_year_map)

    # if the above mapping fails, it results in Nan values in the failed rows,
    # this replaces those with -1
    using_age_class_rows.min_softwood_age = \
        using_age_class_rows.min_softwood_age.fillna(-1)
    using_age_class_rows.min_hardwood_age = \
        using_age_class_rows.min_hardwood_age.fillna(-1)
    using_age_class_rows.max_softwood_age = \
        using_age_class_rows.max_softwood_age.fillna(-1)
    using_age_class_rows.max_hardwood_age = \
        using_age_class_rows.max_hardwood_age.fillna(-1)

    # return the final substituted rows
    result = non_using_age_class_rows.append(using_age_class_rows) \
        .reset_index(drop=True)

    # check that all age criteria are identical between SW and HW (since CBM
    # has only a stand age)
    differing_age_criteria = result.loc[
        (result.min_softwood_age != result.min_hardwood_age) |
        (result.max_softwood_age != result.max_hardwood_age)]
    if len(differing_age_criteria) > 0:
        raise ValueError(
            "Values of column min_softwood_age must equal values of column "
            "min_hardwood_age, and values of column max_softwood_age must "
            "equal values of column max_hardwood_age since CBM defines only "
            "a stand age and does not track hardwood and softwood age "
            "seperately.")

    return result
