import numpy as np
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_classifier_parser


def parse(yield_table, classifiers, classifier_values, age_classes):
    """Parses and validates the CBM SIT growth and yield format.

    Args:
        yield_table (pandas.DataFrame): SIT formatted growth and yield data
        classifiers (pandas.DataFrame): used to validate the classifier
            set columns of the yield data. Use the return value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): used to validate the classifier
            set columns of the yield data. Use the return value of:
            :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        age_classes (pandas.DataFrame): used to validate the number of volume
            columns.  Use the return value of:
            :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Raises:
        ValueError: the specified data did not have the correct number of
            columns according to the defined classifiers and age classes
        ValueError: the leading_species column contained a value that was
            not defined in the specified species map.
        ValueError: Classifier sets were not valid according to the specified
            classifiers and classifier_values.

    Returns:
        pandas.DataFrame: Validated sit input with standardized column names
            and substituted species
    """
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
