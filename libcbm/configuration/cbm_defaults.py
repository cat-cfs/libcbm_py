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
            configuration.
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
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0
        query = \
            libcbm.configuration.cbm_defaults_queries.get_query("pools.sql")
        for row in cursor.execute(query):
            result.append({"name": row[0], "id": row[1], "index": index})
            index += 1
        return result


def load_flux_indicators(sqlitePath):
    result = []
    flux_indicator_source_sql = """
        select flux_indicator_source.pool_id from flux_indicator
        inner join flux_indicator_source on flux_indicator_source.flux_indicator_id = flux_indicator.id
        where flux_indicator.id = ?"""
    flux_indicator_sink_sql = """
        select flux_indicator_sink.pool_id from flux_indicator
        inner join flux_indicator_sink on flux_indicator_sink.flux_indicator_id = flux_indicator.id
        where flux_indicator.id = ?"""
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0
        flux_indicator_rows = list(cursor.execute("select id, flux_process_id from flux_indicator order by id"))
        for row in flux_indicator_rows:
            flux_indicator = {
                "id": row[0],
                "index": index,
                "process_id": row[1],
                "source_pools": [],
                "sink_pools": []
            }
            for source_pool_row in cursor.execute(flux_indicator_source_sql, (row[0],)):
                flux_indicator["source_pools"].append(int(source_pool_row[0]))
            for sink_pool_row in cursor.execute(flux_indicator_sink_sql, (row[0],)):
                flux_indicator["sink_pools"].append(int(sink_pool_row[0]))
            result.append(flux_indicator)
            index += 1
        return result
