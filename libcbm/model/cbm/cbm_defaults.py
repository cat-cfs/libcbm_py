"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import pandas as pd
import sqlite3
from libcbm.model.cbm import cbm_defaults_queries


def load_cbm_parameters(sqlite_path):
    """Loads cbm default parameters into configuration dictionary format.
    Used for initializing CBM functionality in LibCBM via the InitializeCBM
    function.

    Args:
        sqlite_path (str): Path to a CBM parameters database as formatted
            like: https://github.com/cat-cfs/cbm_defaults

    Raises:
        AssertionError:  if the name of any 2 queries is the same, an error is
            raised.

    Returns:
        dict: a dictionary of name/formatted data pairs for use with LibCBM
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

    if not os.path.exists(sqlite_path):
        # sqlite3.connect does not raise an error on no path
        raise ValueError(
            "specified path does not exist '{0}'".format(sqlite_path))
    with sqlite3.connect(sqlite_path) as conn:
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


def parameter_as_dataframe(parameters):
    """Return one of the values stored in the dictionary returned by
    :py:func:`load_cbm_parameters` as a DataFrame

    Args:
        parameters (dict): a dictionary with keys:

          - column_map: a dictionary of name (key) to column index (value)
          - data: list of lists of values in the table
    """
    colmap = parameters["column_map"]
    keys = sorted(colmap, key=colmap.get)
    colnames = []
    for key in keys:
        colnames.append(key)
    return pd.DataFrame(
        data=parameters["data"],
        columns=colnames)


def dataframe_as_parameter(df):
    """Convert a dataframe into a dictionary table format like the
    dictionary values returned by the function
    :py:func:`load_cbm_parameters`

    Args:
        df (pandas.DataFrame): a dataframe to convert to the data/col_map
            scheme
    """
    return {
        "column_map": {x: i for i, x in enumerate(df.columns)},
        "data": df.values.tolist()}


def load_cbm_pools(sqlite_path):
    """Loads cbm pool information from a cbm_defaults database into the
    format expected by the libcbm compiled library.

    Args:
        sqlite_path (str): path to a cbm_defaults database

    Returns:
        list: list of dictionaries describing CBM pools

            For example::

                [
                    {"name": "pool1", "id": 1, "index": 0},
                    {"name": "pool2", "id": 2, "index": 1},
                    ...,
                    {"name": "poolN", "id": N, "index": N-1},
                ]
    """
    result = []
    with sqlite3.connect(sqlite_path) as conn:
        cursor = conn.cursor()
        index = 0
        query = cbm_defaults_queries.get_query("pools.sql")
        for row in cursor.execute(query):
            result.append({"name": row[0], "id": row[1], "index": index})
            index += 1
        return result


def load_cbm_flux_indicators(sqlite_path):
    """Loads cbm flux indicator information from a cbm_defaults database
    into the format expected by the libcbm compiled library.

    Used to capture flows between specified source pools and specified sink
    pools for a given process to return as model output.

    Args:
        sqlite_path (str): path to a cbm_defaults database

    Returns:
        list: list of dictionaries describing CBM flux indicators.

            For example::

                [
                    {
                        "id": 1,
                        "index": 0,
                        "process_id": 1,
                        "source_pools": [1, 2, 3, 4],
                        "sink_pools": [5, 6, 7, 8],
                    },
                ]
    """
    result = []
    flux_indicator_source_sql = cbm_defaults_queries.get_query(
        "flux_indicator_source.sql")
    flux_indicator_sink_sql = cbm_defaults_queries.get_query(
        "flux_indicator_sink.sql")
    with sqlite3.connect(sqlite_path) as conn:
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


def get_cbm_parameters_factory(db_path):
    """Get a function that formates CBM parameters for
    :py:class:`libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper`
    drawn from the specified database.

    Args:
        db_path (str): path to a cbm_defaults database

    Returns:
        func: a function that creates CBM parameters

        Compatible with: :py:func:`libcbm.model.cbm.cbm_factory.create`
    """
    def factory():
        return load_cbm_parameters(db_path)
    return factory


def get_libcbm_configuration_factory(db_path):
    """Get a parameterless function that creates configuration for
    :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        db_path (str): path to a cbm_defaults database

    Returns:
        func: a function that creates CBM configuration input for libcbm

        Compatible with: :py:func:`libcbm.model.cbm.cbm_factory.create`
    """
    def factory():
        return {
            "pools": load_cbm_pools(db_path),
            "flux_indicators": load_cbm_flux_indicators(db_path)
        }
    return factory
