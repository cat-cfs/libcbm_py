import numpy as np
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_classifier_parser


def parse(yield_table, classifiers, classifier_values, age_classes,
          species_map):

    yield_format = sit_format.get_yield_format(
        classifiers.name, len(yield_table.columns))

    unpacked_table = sit_parser.unpack_table(
        yield_table, yield_format, "yield")

    # check that the number of volumes is equal to the number of age classes
    expected_column_count = len(age_classes) + len(classifiers) + 1
    if expected_column_count != len(unpacked_table.columns):
        raise ValueError(
            f"expected {expected_column_count} columns. This is defined as "
            f"{len(classifiers) + 1} classifiers plus {len(age_classes)} "
            "age classes")

    # check that the leading species column is mapped
    undefined_species = set(
        unpacked_table.leading_species.unique()).difference(species_map.keys())
    if undefined_species:
        raise ValueError(
            "the following values in the leading_species column were not "
            f"present in the specified map: {undefined_species}")
    unpacked_table.leading_species = unpacked_table.leading_species.map(
        species_map)

    # check that the correct number of classifiers are present and check that
    # each value in yield table classifier sets is defined in classifier values
    for row in classifiers.itertuples():
        yield_classifiers = unpacked_table[row.name].unique()
        defined_classifier_values = classifier_values[
            classifier_values["classifier_id"] == row.id]["name"].unique()
        wildcard = np.array([sit_classifier_parser.get_wildcard_keyword()])
        valid_classifiers = np.concatenate(
            [defined_classifier_values, wildcard])

        diff = np.setdiff1d(yield_classifiers, valid_classifiers)
        if len(diff) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff}")

    return unpacked_table
