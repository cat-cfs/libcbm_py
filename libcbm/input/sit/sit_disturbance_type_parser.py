
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_parser


def parse_disturbance_types(disturbance_types_table):
    return sit_parser.unpack_table(
        disturbance_types_table,
        sit_format.get_disturbance_type_format(
            len(disturbance_types_table.columns)),
        "disturbance types")
