import pandas as pd
import numpy as np

from libcbm.input.sit import sit_format


def unpack_column(table, column_description, table_name):
    data = table.iloc[:, column_description["index"]]
    col_name = column_description["name"]
    if "type" in column_description:
        data = data.astype(column_description["type"])
    if "min_value" in column_description:
        min_value = column_description["min_value"]
        if len(data[data < min_value]):
            raise ValueError(
                f"{table_name} table, column: '{col_name}' contains values "
                f"less than the minimum allowed value: {min_value}")
    if "max_value" in column_description:
        max_value = column_description["max_value"]
        if len(data[data > max_value]):
            raise ValueError(
                f"{table_name} table, column: '{col_name}' contains values "
                f"greater than the maximum allowed value: {max_value}")
    return data


def unpack_table(table, column_descriptions, table_name):
    cols = [x["name"] for x in column_descriptions]
    data = {
        x["name"]: unpack_column(table, x, table_name)
        for x in column_descriptions}
    return pd.DataFrame(columns=cols, data=data)


def try_get_int(s):
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
            int_x, success = try_get_int(str_x)
            if(success):
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
