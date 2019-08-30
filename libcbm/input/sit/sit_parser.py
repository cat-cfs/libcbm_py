import pandas as pd
import numpy as np

from libcbm.input.sit import sit_format


def unpack_column(table, column_description, table_name):
    data = table.iloc[:, column_description["index"]]
    col_name = column_description["name"]
    if "type" in column_description:
        try:
            data = data.astype(column_description['type'])
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
        return (i, True)
    except ValueError:
        return (None, False)


def get_parse_bool_func(table_name, colname):
    def parse_bool(x):
        if isinstance(x, bool):
            return x
        elif isinstance(x, int):
            return x != 0
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
