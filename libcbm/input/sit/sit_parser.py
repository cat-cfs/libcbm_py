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


def parse_age_classes(age_class_table):
    return unpack_table(
        age_class_table, sit_format.get_age_class_format(),
        "age classes")


def parse_disturbance_types(disturbance_types_table):
    return unpack_table(
        disturbance_types_table,
        sit_format.get_disturbance_type_format(
            len(disturbance_types_table.columns)),
        "disturbance types")

