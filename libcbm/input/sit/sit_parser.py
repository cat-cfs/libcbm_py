import pandas as pd

from libcbm.input.sit import sit_format


def unpack_column(table, column_description):
    data = table[:, column_description["index"]]
    if "type" in column_description["type"]:
        data = data.as_type(column_description["type"])

    return data


def unpack_table(table, column_descriptions):
    cols = [x["name"] for x in column_descriptions]
    data = {
        x["name"]: unpack_column(table, x)
        for x in column_descriptions}
    return pd.DataFrame(columns=cols, data=data)


def parse_age_classes(age_class_table):
    return unpack_table(
        age_class_table, sit_format.get_age_class_format())


def parse_classifiers(classifiers_table):
    classifiers_format = sit_format.get_age_class_format(
        len(classifiers_table.columns))
    unpacked = unpack_table(
        classifiers_table, classifiers_format
    )

