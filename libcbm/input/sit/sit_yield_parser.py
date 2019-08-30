from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def parse(yield_table, classifiers, classifier_values, age_classes,
          species_map):

    yield_format = sit_format.get_yield_format(
        classifiers.name, len(yield_table.columns))

    unpacked_table = sit_parser.unpack_table(
        yield_table, yield_format, "yield")

    # check that the number of volumes is equal to the number of age classes

    # check that the leading species column is mapped

    # check that the correct number of classifiers are present

    # check that each value in yield table classifier sets is defined in
    # classifier values

    return unpacked_table
