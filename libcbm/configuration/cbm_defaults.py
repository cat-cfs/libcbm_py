import sqlite3
import libcbm.configuration.cbm_defaults_queries


def load_cbm_parameters(sqlitePath):
    """Loads cbm default parameters into configuration dictionary format.
    Used for initializing CBM functionality in LibCBM via the InitializeCBM
    function.

    Arguments:
        sqlitePath {str} -- Path to a CBM parameters database as formatted
        like: https://github.com/cat-cfs/cbm_defaults

    Raises:
        AssertionError: if the name of any 2 queries is the same, an error is
            raised.

    Returns:
        dict -- a dictionary of name/formatted data pairs for use with LibCBM
            configuration
    """
    result = {}

    queries = {
        k: libcbm.configuration.cbm_defaults_queries.get_query(
            "{}.sql".format(k))
        for k in [
            "decay_parameters",
            "slow_mixing_rate",
            "mean_annual_temp",
            "turnover_parameters",
            "disturbance_matrix_values",
            "disturbance_matrix_associations",
            "root_parameter",
            "growth_multipliers",
            "land_classes",
            "land_class_transitions",
            "spatial_units",
            "random_return_interval",
            "flux_indicator_process",
            "flux_indicator_source",
            "flux_indicator_sink",
            "spinup_parameter"
            ]}

    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for table, query in queries.items():
            cursor.execute(query)
            data = [[col for col in row] for row in cursor]
            if table in result:
                raise AssertionError(
                    "duplicate table name detected {}"
                    .format(table))
            result[table] = {
                "column_map": {
                    v[0]: i for i, v in
                    enumerate(cursor.description)},
                "data": data
            }

    return result
