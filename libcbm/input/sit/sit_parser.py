import pandas as pd
import numpy as np


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
        return i, True
    except ValueError:
        return None, False


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



def substitute_using_age_class_rows(rows, parse_bool_func, age_classes):
    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output
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
        undefined_age_classes = np.setdiff1d(
            using_age_class_rows[age_class_criteria_col].astype(np.str).unique(),
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
    return result
