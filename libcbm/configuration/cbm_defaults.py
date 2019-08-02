import sqlite3
from libcbm.configuration import cbm_defaults_queries


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
            configuration.
    """
    result = {}

    queries = {
        k: cbm_defaults_queries.get_query(
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
            "spinup_parameter",
            "afforestation_pre_type"
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


def load_cbm_pools(sqlitePath):
    """Loads cbm pool information from a cbm_defaults database into the
    format expected by the libcbm compiled library.

    Example of output:
        [
            {"name": "pool1", "id": 1, "index": 0},
            {"name": "pool2", "id": 2, "index": 1},
            ...,
            {"name": "poolN", "id": N, "index": N-1},
        ]
    Arguments:
        sqlitePath {str} -- path to a cbm_defaults database

    Returns:
        list -- list of dictionaries describing CBM pools
    """
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0
        query = cbm_defaults_queries.get_query("pools.sql")
        for row in cursor.execute(query):
            result.append({"name": row[0], "id": row[1], "index": index})
            index += 1
        return result


def load_flux_indicators(sqlitePath):
    """Loads cbm flux indicator information from a cbm_defaults database
    into the format expected by the libcbm compiled library.

    Used to capture flows between specified source pools and specified sink
    pools for a given process to return as model output.

    Example of output:
        [
            {
                "id": 1,
                "index": 0,
                "process_id": 1,
                "source_pools": [1, 2, 3, 4],
                "sink_pools": [5, 6, 7, 8],
            },
        ]

    Arguments:
        sqlitePath {str} -- path to a cbm_defaults database

    Returns:
        list -- list of dictionaries describing CBM flux indicators
    """
    result = []
    flux_indicator_source_sql = cbm_defaults_queries.get_query(
        "flux_indicator_source.sql")
    flux_indicator_sink_sql = cbm_defaults_queries.get_query(
        "flux_indicator_sink.sql")
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0
        flux_indicator_sql = cbm_defaults_queries.get_query(
            "flux_indicator.sql")
        flux_indicator_rows = list(cursor.execute(flux_indicator_sql))
        for row in flux_indicator_rows:
            flux_indicator = {
                "id": row[0],
                "index": index,
                "process_id": row[1],
                "source_pools": [],
                "sink_pools": []
            }
            for source_pool_row in cursor.execute(
                    flux_indicator_source_sql, (row[0],)):
                flux_indicator["source_pools"].append(int(source_pool_row[0]))
            for sink_pool_row in cursor.execute(
                    flux_indicator_sink_sql, (row[0],)):
                flux_indicator["sink_pools"].append(int(sink_pool_row[0]))
            result.append(flux_indicator)
            index += 1
        return result
